print("App starting...")
import os
import re
import json
import uuid
import time
import shutil
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import mlflow
import pdfplumber
from pypdf import PdfReader
from dotenv import load_dotenv

try:
    from databricks_openai import DatabricksOpenAI
    DATABRICKS_OPENAI_AVAILABLE = True
except Exception:
    DATABRICKS_OPENAI_AVAILABLE = False

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        TimestampType,
    )
    SPARK_AVAILABLE = True
except Exception:
    SPARK_AVAILABLE = False


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("PDFExtractionStreamlitApp")


DEFAULT_CONFIG = {
    "catalog": os.getenv("CATALOG", "structured_data"),
    "schema": os.getenv("SCHEMA", "extraction"),
    "volume": os.getenv("VOLUME", "uploaded_pdf_files"),
    "pdf_volume_path": os.getenv(
        "PDF_VOLUME_PATH",
        "/Volumes/structured_data/extraction/documents/uploaded_pdf_files",
    ),
    "results_table": os.getenv(
        "RESULTS_TABLE",
        "pdf_extractions_normalized",
    ),
    "log_table": os.getenv(
        "LOG_TABLE",
        "pdf_extraction_run_log",
    ),
    "model": os.getenv(
        "MODEL_NAME",
        "databricks-meta-llama-3-3-70b-instruct",
    ),
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "databricks"),
    "mlflow_experiment": os.getenv(
        "MLFLOW_EXPERIMENT",
        "/Users/example/pdf-extraction-agent",
    ),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "10000")),
    "databricks_host": os.getenv("DATABRICKS_HOST", ""),
    "databricks_token": os.getenv("DATABRICKS_TOKEN", ""),
    "local_volume_fallback": os.getenv(
        "LOCAL_VOLUME_FALLBACK",
        "./local_volume_store",
    ),
    "enable_delta_write": os.getenv(
        "ENABLE_DELTA_WRITE",
        "false",
    ).lower() == "true",
    "enable_mlflow": os.getenv(
        "ENABLE_MLFLOW",
        "false",
    ).lower() == "true",
}

EXTRACTION_FIELDS = [
    "module_names",
    "instructor_names",
    "dates",
    "emails",
    "attention_models",
    "urls",
]


def get_spark():
    if not SPARK_AVAILABLE:
        return None
    try:
        return SparkSession.builder.getOrCreate()
    except Exception as exc:
        log.warning(f"Spark unavailable: {exc}")
        return None


def get_dbutils():
    try:
        from pyspark.dbutils import DBUtils

        spark = get_spark()
        if spark is not None:
            return DBUtils(spark)
    except Exception as exc:
        log.warning(f"dbutils unavailable: {exc}")
    return None


def set_mlflow():
    try:
        mlflow.set_tracking_uri(DEFAULT_CONFIG["mlflow_tracking_uri"])
        if DEFAULT_CONFIG["mlflow_experiment"]:
            mlflow.set_experiment(DEFAULT_CONFIG["mlflow_experiment"])
        return True
    except Exception as exc:
        log.warning(f"MLflow setup skipped: {exc}")
        return False


def sanitize_filename(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return base or f"file_{uuid.uuid4().hex}.pdf"


def ensure_local_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def best_effort_create_uc_objects(spark, config: dict[str, Any]):
    if spark is None:
        return

    try:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS `{config['catalog']}`")
        spark.sql(
            f"CREATE SCHEMA IF NOT EXISTS `{config['catalog']}`.`{config['schema']}`"
        )
    except Exception as exc:
        log.warning(f"Catalog schema creation skipped: {exc}")

    try:
        volume_name = config["volume"]
        spark.sql(
            f"CREATE VOLUME IF NOT EXISTS `{config['catalog']}`.`{config['schema']}`.`{volume_name}`"
        )
    except Exception as exc:
        log.warning(f"Volume creation skipped: {exc}")

    try:
        full_results = (
            f"`{config['catalog']}`.`{config['schema']}`.`{config['results_table']}`"
        )
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {full_results} (
                extraction_id STRING NOT NULL,
                run_id STRING,
                file_name STRING,
                pdf_path STRING,
                parameter STRING,
                value STRING,
                citation_page STRING,
                citation_text STRING,
                citation_url STRING,
                status STRING,
                model_used STRING,
                created_at TIMESTAMP,
                raw_metadata STRING
            )
            USING DELTA
            """
        )
    except Exception as exc:
        log.warning(f"Results table creation skipped: {exc}")

    try:
        full_log = f"`{config['catalog']}`.`{config['schema']}`.`{config['log_table']}`"
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {full_log} (
                run_id STRING NOT NULL,
                extraction_id STRING,
                file_name STRING,
                pdf_path STRING,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                status STRING,
                error_msg STRING,
                raw_json STRING
            )
            USING DELTA
            """
        )
    except Exception as exc:
        log.warning(f"Log table creation skipped: {exc}")


class VolumeManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.spark = get_spark()
        self.dbutils = get_dbutils()
        self.use_uc_volume = self.spark is not None and self.dbutils is not None

        if self.spark is not None:
            best_effort_create_uc_objects(self.spark, config)

        ensure_local_dir(self.config["local_volume_fallback"])

    def save_uploaded_file(self, uploaded_file) -> dict[str, str]:
        safe_name = sanitize_filename(uploaded_file.name)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_name = f"{stamp}_{uuid.uuid4().hex[:8]}_{safe_name}"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        if self.use_uc_volume:
            volume_path = f"{self.config['pdf_volume_path'].rstrip('/')}/{unique_name}"
            try:
                self.dbutils.fs.cp(f"file:{temp_path}", volume_path)
                return {
                    "file_name": uploaded_file.name,
                    "local_temp_path": temp_path,
                    "stored_pdf_path": volume_path,
                    "citation_url": self._build_citation_url(volume_path),
                }
            except Exception as exc:
                log.warning(f"Volume save failed, using local fallback: {exc}")

        local_path = os.path.join(self.config["local_volume_fallback"], unique_name)
        shutil.copy(temp_path, local_path)
        return {
            "file_name": uploaded_file.name,
            "local_temp_path": temp_path,
            "stored_pdf_path": local_path,
            "citation_url": Path(local_path).resolve().as_uri(),
        }

    def _build_citation_url(self, pdf_path: str) -> str:
        return pdf_path


class PDFTextExtractor:
    def extract(self, pdf_path: str) -> dict[str, Any]:
        pages = []
        full_text_parts = []
        metadata = {}

        try:
            reader = PdfReader(pdf_path)
            metadata = dict(reader.metadata or {})
        except Exception:
            metadata = {}

        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append(
                    {
                        "page_number": idx,
                        "text": text,
                    }
                )
                full_text_parts.append(f"\n[PAGE {idx}]\n{text}")

        return {
            "full_text": "\n".join(full_text_parts),
            "pages": pages,
            "metadata": metadata,
            "page_count": len(pages),
        }


class RegexExtractor:
    def extract(self, text: str) -> dict[str, list[str]]:
        patterns = {
            "module_names": r"Module\s+\d+[:\-]?\s*([A-Za-z0-9 ,&\-\(\)]+)",
            "instructor_names": r"Instructor[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            "dates": r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b",
            "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "attention_models": r"\bTransformer\b|\bBERT\b|\bGPT\b|\bLLaMA\b|\bAttention\b",
            "urls": r"https?://[^\s)>\]]+",
        }

        results = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            values = []

            for match in matches:
                if isinstance(match, tuple):
                    values.append(" ".join(str(x) for x in match if x))
                else:
                    values.append(str(match))

            results[key] = sorted({v.strip() for v in values if v and v.strip()})

        return results


class LLMExtractor:
    def __init__(self, model: str, chunk_size: int = 10000):
        self.model = model
        self.chunk_size = chunk_size
        self.client = None

        if DATABRICKS_OPENAI_AVAILABLE:
            try:
                self.client = DatabricksOpenAI()
            except Exception as exc:
                log.warning(f"DatabricksOpenAI init failed: {exc}")

    def _chunk_text(self, text: str) -> list[str]:
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def extract(self, text: str) -> dict[str, list[str]]:
        aggregate = {k: [] for k in EXTRACTION_FIELDS}

        if not self.client:
            return aggregate

        for chunk in self._chunk_text(text):
            prompt = f"""
Extract the following from the text and return strict JSON only.

Fields:
- module_names
- instructor_names
- dates
- emails
- attention_models
- urls

Return format:
{{
  "module_names": [],
  "instructor_names": [],
  "dates": [],
  "emails": [],
  "attention_models": [],
  "urls": []
}}

Text:
{chunk}
"""
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Return only valid JSON. No explanation.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0,
                )

                content = response.choices[0].message.content.strip()
                parsed = json.loads(content)

                for key in EXTRACTION_FIELDS:
                    values = parsed.get(key, [])
                    if isinstance(values, list):
                        aggregate[key].extend(
                            str(v).strip() for v in values if str(v).strip()
                        )
            except Exception as exc:
                log.warning(f"LLM extraction chunk skipped: {exc}")

        for key in aggregate:
            aggregate[key] = sorted({v for v in aggregate[key] if v})

        return aggregate


class ResultMerger:
    @staticmethod
    def merge(
        regex_results: dict[str, list[str]],
        llm_results: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        merged = {}
        for key in EXTRACTION_FIELDS:
            values = regex_results.get(key, []) + llm_results.get(key, [])
            merged[key] = sorted({str(v).strip() for v in values if str(v).strip()})
        return merged


class CitationResolver:
    @staticmethod
    def find_citation(value: str, pages: list[dict[str, Any]]) -> tuple[str, str]:
        value_lower = value.lower()

        for page in pages:
            text = page["text"] or ""
            idx = text.lower().find(value_lower)
            if idx >= 0:
                start = max(0, idx - 120)
                end = min(len(text), idx + len(value) + 120)
                snippet = text[start:end].replace("\n", " ").strip()
                return str(page["page_number"]), snippet

        return "", ""

    def to_rows(
        self,
        extraction_id: str,
        run_id: str,
        file_name: str,
        stored_pdf_path: str,
        citation_base_url: str,
        model: str,
        metadata: dict[str, Any],
        results: dict[str, list[str]],
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = []
        created_at = datetime.now(timezone.utc)

        for field in EXTRACTION_FIELDS:
            values = results.get(field, [])

            if not values:
                rows.append(
                    {
                        "extraction_id": extraction_id,
                        "run_id": run_id,
                        "file_name": file_name,
                        "pdf_path": stored_pdf_path,
                        "parameter": field,
                        "value": "",
                        "citation_page": "",
                        "citation_text": "",
                        "citation_url": "",
                        "status": "Not Found",
                        "model_used": model,
                        "created_at": created_at,
                        "raw_metadata": json.dumps(metadata or {}),
                    }
                )
                continue

            for value in values:
                citation_page, citation_text = self.find_citation(value, pages)
                citation_url = ""

                if citation_base_url and citation_page:
                    citation_url = f"{citation_base_url}#page={citation_page}"

                rows.append(
                    {
                        "extraction_id": extraction_id,
                        "run_id": run_id,
                        "file_name": file_name,
                        "pdf_path": stored_pdf_path,
                        "parameter": field,
                        "value": value,
                        "citation_page": citation_page,
                        "citation_text": citation_text,
                        "citation_url": citation_url,
                        "status": "Success" if value else "Not Found",
                        "model_used": model,
                        "created_at": created_at,
                        "raw_metadata": json.dumps(metadata or {}),
                    }
                )

        return rows


class DeltaWriter:
    RESULT_SCHEMA = (
        StructType(
            [
                StructField("extraction_id", StringType(), False),
                StructField("run_id", StringType(), True),
                StructField("file_name", StringType(), True),
                StructField("pdf_path", StringType(), True),
                StructField("parameter", StringType(), True),
                StructField("value", StringType(), True),
                StructField("citation_page", StringType(), True),
                StructField("citation_text", StringType(), True),
                StructField("citation_url", StringType(), True),
                StructField("status", StringType(), True),
                StructField("model_used", StringType(), True),
                StructField("created_at", TimestampType(), True),
                StructField("raw_metadata", StringType(), True),
            ]
        )
        if SPARK_AVAILABLE
        else None
    )

    LOG_SCHEMA = (
        StructType(
            [
                StructField("run_id", StringType(), False),
                StructField("extraction_id", StringType(), True),
                StructField("file_name", StringType(), True),
                StructField("pdf_path", StringType(), True),
                StructField("started_at", TimestampType(), True),
                StructField("finished_at", TimestampType(), True),
                StructField("status", StringType(), True),
                StructField("error_msg", StringType(), True),
                StructField("raw_json", StringType(), True),
            ]
        )
        if SPARK_AVAILABLE
        else None
    )

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.spark = get_spark()

    def write_results(self, rows: list[dict[str, Any]]):
        if not self.spark or not self.config["enable_delta_write"] or not rows:
            return

        try:
            full_table = (
                f"{self.config['catalog']}."
                f"{self.config['schema']}."
                f"{self.config['results_table']}"
            )
            df = self.spark.createDataFrame(rows, schema=self.RESULT_SCHEMA)
            df.write.format("delta").mode("append").saveAsTable(full_table)
        except Exception as exc:
            log.warning(f"Delta result write skipped: {exc}")

    def write_log(
        self,
        run_id: str,
        extraction_id: str,
        file_name: str,
        pdf_path: str,
        started_at: datetime,
        finished_at: datetime,
        status: str,
        results: dict | None = None,
        error_msg: str = "",
    ):
        if not self.spark or not self.config["enable_delta_write"]:
            return

        try:
            full_table = (
                f"{self.config['catalog']}."
                f"{self.config['schema']}."
                f"{self.config['log_table']}"
            )
            row = [
                {
                    "run_id": run_id,
                    "extraction_id": extraction_id,
                    "file_name": file_name,
                    "pdf_path": pdf_path,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "status": status,
                    "error_msg": error_msg,
                    "raw_json": json.dumps(results or {}),
                }
            ]
            df = self.spark.createDataFrame(row, schema=self.LOG_SCHEMA)
            df.write.format("delta").mode("append").saveAsTable(full_table)
        except Exception as exc:
            log.warning(f"Delta log write skipped: {exc}")


class PDFExtractionRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.volume = VolumeManager(config)
        self.extractor = PDFTextExtractor()
        self.regex = RegexExtractor()
        self.llm = LLMExtractor(config["model"], config["chunk_size"])
        self.merger = ResultMerger()
        self.citation = CitationResolver()
        self.writer = DeltaWriter(config)

    def run_single(self, uploaded_file) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        extraction_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        stored = self.volume.save_uploaded_file(uploaded_file)
        file_name = stored["file_name"]
        local_temp_path = stored["local_temp_path"]
        stored_pdf_path = stored["stored_pdf_path"]
        citation_base_url = stored["citation_url"]

        try:
            pdf_data = self.extractor.extract(local_temp_path)
            full_text = pdf_data["full_text"]

            if not full_text.strip():
                raise ValueError("No text extracted from PDF.")

            regex_results = self.regex.extract(full_text)
            llm_results = self.llm.extract(full_text)
            merged = self.merger.merge(regex_results, llm_results)

            rows = self.citation.to_rows(
                extraction_id=extraction_id,
                run_id=run_id,
                file_name=file_name,
                stored_pdf_path=stored_pdf_path,
                citation_base_url=citation_base_url,
                model=self.config["model"],
                metadata={
                    "page_count": pdf_data["page_count"],
                    "info": pdf_data["metadata"],
                },
                results=merged,
                pages=pdf_data["pages"],
            )

            finished_at = datetime.now(timezone.utc)

            self.writer.write_results(rows)
            self.writer.write_log(
                run_id=run_id,
                extraction_id=extraction_id,
                file_name=file_name,
                pdf_path=stored_pdf_path,
                started_at=started_at,
                finished_at=finished_at,
                status="success",
                results={"rows": rows},
            )

            return {
                "status": "success",
                "run_id": run_id,
                "extraction_id": extraction_id,
                "file_name": file_name,
                "stored_pdf_path": stored_pdf_path,
                "rows": rows,
                "counts": {k: len(merged.get(k, [])) for k in EXTRACTION_FIELDS},
            }

        except Exception as exc:
            finished_at = datetime.now(timezone.utc)

            self.writer.write_log(
                run_id=run_id,
                extraction_id=extraction_id,
                file_name=file_name,
                pdf_path=stored_pdf_path,
                started_at=started_at,
                finished_at=finished_at,
                status="failed",
                error_msg=str(exc),
            )

            return {
                "status": "failed",
                "run_id": run_id,
                "extraction_id": extraction_id,
                "file_name": file_name,
                "stored_pdf_path": stored_pdf_path,
                "rows": [],
                "error": str(exc),
            }

        finally:
            try:
                os.remove(local_temp_path)
            except Exception:
                pass


def render_clickable_table(df: pd.DataFrame):
    render_df = df.copy()

    render_df["citation"] = render_df.apply(
        lambda row: (
            f'<a href="{row["citation_url"]}" target="_blank">'
            f'Open PDF page {row["citation_page"]}</a>'
            if str(row.get("citation_url", "")).strip()
            and str(row.get("citation_page", "")).strip()
            else ""
        ),
        axis=1,
    )

    columns = [
        "file_name",
        "parameter",
        "value",
        "citation_page",
        "citation_text",
        "citation",
        "status",
    ]

    html = render_df[columns].to_html(escape=False, index=False)
    st.markdown(html, unsafe_allow_html=True)


def sidebar_config() -> dict[str, Any]:
    st.sidebar.header("Configuration")

    return {
        "catalog": st.sidebar.text_input(
            "Catalog",
            value=DEFAULT_CONFIG["catalog"],
        ),
        "schema": st.sidebar.text_input(
            "Schema",
            value=DEFAULT_CONFIG["schema"],
        ),
        "volume": st.sidebar.text_input(
            "Volume",
            value=DEFAULT_CONFIG["volume"],
        ),
        "pdf_volume_path": st.sidebar.text_input(
            "PDF Volume Path",
            value=DEFAULT_CONFIG["pdf_volume_path"],
        ),
        "results_table": st.sidebar.text_input(
            "Results Table",
            value=DEFAULT_CONFIG["results_table"],
        ),
        "log_table": st.sidebar.text_input(
            "Log Table",
            value=DEFAULT_CONFIG["log_table"],
        ),
        "model": st.sidebar.text_input(
            "Foundation Model",
            value=DEFAULT_CONFIG["model"],
        ),
        "mlflow_tracking_uri": st.sidebar.text_input(
            "MLflow Tracking URI",
            value=DEFAULT_CONFIG["mlflow_tracking_uri"],
        ),
        "mlflow_experiment": st.sidebar.text_input(
            "MLflow Experiment",
            value=DEFAULT_CONFIG["mlflow_experiment"],
        ),
        "chunk_size": int(
            st.sidebar.number_input(
                "Chunk Size",
                min_value=1000,
                max_value=50000,
                value=DEFAULT_CONFIG["chunk_size"],
                step=1000,
            )
        ),
        "databricks_host": DEFAULT_CONFIG["databricks_host"],
        "databricks_token": DEFAULT_CONFIG["databricks_token"],
        "local_volume_fallback": DEFAULT_CONFIG["local_volume_fallback"],
        "enable_delta_write": st.sidebar.checkbox(
            "Enable Delta Writes",
            value=DEFAULT_CONFIG["enable_delta_write"],
        ),
        "enable_mlflow": st.sidebar.checkbox(
            "Enable MLflow",
            value=DEFAULT_CONFIG["enable_mlflow"],
        ),
    }


def main():
    st.set_page_config(
        page_title="PDF Extraction Agent POC",
        layout="wide",
    )

    st.title("PDF Extraction Agent POC")
    st.write(
        "Upload one or more PDFs, run extraction, and review results with citations."
    )

    config = sidebar_config()

    st.info(
        "POC note: In some environments, direct browser openable Volume URLs may be "
        "limited. This app stores uploaded PDFs in a configured Volume when available, "
        "and falls back to local storage otherwise. Citations are clickable using the "
        "best practical path available for the running environment."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF file or files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    run_button = st.button("Run Agent", type="primary")

    if not run_button:
        return

    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        st.stop()

    if config["enable_mlflow"]:
        set_mlflow()

    runner = PDFExtractionRunner(config)
    all_rows = []
    success_files = []
    failed_files = []

    status_box = st.empty()
    progress_bar = st.progress(0)

    if config["enable_mlflow"]:
        try:
            mlflow.start_run(
                run_name=f"streamlit_pdf_extraction_{int(time.time())}"
            )
            mlflow.log_params(
                {
                    "catalog": config["catalog"],
                    "schema": config["schema"],
                    "volume": config["volume"],
                    "pdf_volume_path": config["pdf_volume_path"],
                    "results_table": config["results_table"],
                    "log_table": config["log_table"],
                    "model": config["model"],
                    "chunk_size": config["chunk_size"],
                }
            )
        except Exception as exc:
            log.warning(f"MLflow run start skipped: {exc}")

    try:
        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            status_box.info(f"Processing {uploaded_file.name} ...")
            result = runner.run_single(uploaded_file)

            if result["status"] == "success":
                success_files.append(result["file_name"])
                all_rows.extend(result["rows"])
            else:
                failed_files.append(
                    (
                        result["file_name"],
                        result.get("error", "Unknown error"),
                    )
                )

            progress_bar.progress(idx / len(uploaded_files))

        status_box.empty()

        if success_files and not failed_files:
            st.success(
                f"Agent run completed successfully for {len(success_files)} file(s)."
            )
        elif success_files and failed_files:
            st.warning(
                "Agent run completed with partial success. "
                f"Success: {len(success_files)}, Failed: {len(failed_files)}"
            )
        else:
            st.error("Agent run failed for all files.")

        if failed_files:
            with st.expander("Failed Files"):
                for file_name, error in failed_files:
                    st.write(f"- {file_name}: {error}")

        if all_rows:
            df = pd.DataFrame(all_rows)

            st.subheader("Extraction Results")
            render_clickable_table(df)

            st.subheader("Raw Results")
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results as CSV",
                data=csv_data,
                file_name="pdf_extraction_results.csv",
                mime="text/csv",
            )

            if config["enable_mlflow"] and mlflow.active_run():
                try:
                    mlflow.log_metric(
                        "processed_file_count",
                        len(uploaded_files),
                    )
                    mlflow.log_metric(
                        "success_file_count",
                        len(success_files),
                    )
                    mlflow.log_metric(
                        "failed_file_count",
                        len(failed_files),
                    )

                    for field in EXTRACTION_FIELDS:
                        mlflow.log_metric(
                            f"{field}_count",
                            int((df["parameter"] == field).sum()),
                        )
                except Exception as exc:
                    log.warning(f"MLflow metrics skipped: {exc}")

    finally:
        if config["enable_mlflow"] and mlflow.active_run():
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    main()