from datetime import date, datetime, timedelta
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

st.set_page_config(page_title="Pillcare | Medicine companion", page_icon="💊", layout="wide")

try:
    engine = create_engine("sqlite:///pillcare.db")
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS medicines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine TEXT NOT NULL,
                dose TEXT NOT NULL,
                time TEXT NOT NULL,
                taken BOOLEAN DEFAULT 0,
                refill INTEGER DEFAULT 0
            )
        """))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS dose_occurrences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_id INTEGER NOT NULL REFERENCES medicines(id) ON DELETE CASCADE,
                scheduled_date DATE NOT NULL,
                scheduled_time TEXT NOT NULL,
                medicine_name_snapshot TEXT,
                dose_snapshot TEXT,
                status TEXT NOT NULL DEFAULT 'Pending'
                    CHECK (status IN ('Pending', 'Taken')),
                taken_at DATETIME,
                UNIQUE (medicine_id, scheduled_date)
            )
        """))
        occurrence_columns = {
            column["name"]
            for column in connection.execute(
                text("PRAGMA table_info(dose_occurrences)")
            ).mappings()
        }
        if "medicine_name_snapshot" not in occurrence_columns:
            connection.execute(
                text(
                    "ALTER TABLE dose_occurrences "
                    "ADD COLUMN medicine_name_snapshot TEXT"
                )
            )
        if "dose_snapshot" not in occurrence_columns:
            connection.execute(
                text("ALTER TABLE dose_occurrences ADD COLUMN dose_snapshot TEXT")
            )
        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_dose_occurrences_scheduled_date
            ON dose_occurrences (scheduled_date)
        """))
except SQLAlchemyError:
    st.error(
        "Unable to open the PillCare database. Check that pillcare.db is accessible "
        "and is a valid SQLite database."
    )
    st.stop()

load_dotenv()

TIME_PATTERN = re.compile(r"^(0?[1-9]|1[0-2]):[0-5]\d\s?(AM|PM)$", re.IGNORECASE)
DEMO_MEDICINES = pd.DataFrame(
    [
        {"Medicine": "Vitamin D3", "Dose": "1000 IU", "Time": "8:00 AM", "Taken": True, "Refill": 24},
        {"Medicine": "Metformin", "Dose": "500 mg", "Time": "9:00 AM", "Taken": False, "Refill": 8},
        {"Medicine": "Omega-3", "Dose": "1 capsule", "Time": "1:00 PM", "Taken": False, "Refill": 16},
        {"Medicine": "Cetirizine", "Dose": "10 mg", "Time": "8:00 PM", "Taken": False, "Refill": 3},
    ]
)


def parse_medicine_time(value):
    value = str(value).strip().upper()
    if not TIME_PATTERN.fullmatch(value):
        return None
    try:
        return datetime.strptime(value, "%I:%M %p").time()
    except ValueError:
        return datetime.strptime(value, "%I:%M%p").time()


def normalized_time(value):
    parsed_time = parse_medicine_time(value)
    if parsed_time is None:
        return str(value).strip().casefold()
    return parsed_time.strftime("%I:%M %p").lstrip("0").casefold()


def search_rxnorm_medicine(medicine_name):
    def fetch_json(endpoint, params=None):
        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=(3.05, 10),
            )
        except requests.Timeout:
            return "timeout", None
        except requests.RequestException:
            return "unavailable", None

        if response.status_code == 429:
            return "rate_limit", None
        try:
            response.raise_for_status()
        except requests.RequestException:
            return "unavailable", None
        try:
            payload = response.json()
        except ValueError:
            return "invalid_response", None
        return ("response", payload) if isinstance(payload, dict) else ("invalid_response", None)

    status, payload = fetch_json(
        "https://rxnav.nlm.nih.gov/REST/rxcui.json",
        {"name": medicine_name, "search": 2, "allsrc": 0},
    )
    if status != "response":
        return status, []

    id_group = payload.get("idGroup")
    if not isinstance(id_group, dict):
        return "invalid_response", []
    rxnorm_ids = id_group.get("rxnormId", [])
    if rxnorm_ids is None:
        return "not_found", []
    if not isinstance(rxnorm_ids, list) or not all(
        isinstance(rxnorm_id, str) and rxnorm_id for rxnorm_id in rxnorm_ids
    ):
        return "invalid_response", []
    if not rxnorm_ids:
        return "not_found", []

    results = []
    for rxnorm_id in rxnorm_ids:
        status, payload = fetch_json(
            f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxnorm_id}/properties.json"
        )
        if status != "response":
            return status, []
        properties = payload.get("properties")
        if not isinstance(properties, dict):
            return "invalid_response", []
        result = {
            label: properties[field]
            for label, field in (
                ("Name", "name"),
                ("Synonym", "synonym"),
                ("RxNorm ID", "rxcui"),
                ("Term type", "tty"),
            )
            if isinstance(properties.get(field), str) and properties[field].strip()
        }
        if result:
            results.append(result)

    return ("results", results) if results else ("missing_fields", [])


def validate_medicine(name, dose, medicine_time, refill):
    if not name.strip():
        return None, "Medicine name cannot be blank."
    if not dose.strip():
        return None, "Dose cannot be blank."
    parsed_time = parse_medicine_time(medicine_time)
    if parsed_time is None:
        return None, "Enter a valid time, such as 9:00 AM."
    if refill < 0:
        return None, "Supply left cannot be negative."
    return (
        {
            "medicine": name.strip(),
            "dose": dose.strip(),
            "time": parsed_time.strftime("%I:%M %p").lstrip("0"),
            "refill": int(refill),
        },
        None,
    )


def load_medicines():
    try:
        with engine.connect() as connection:
            medicines = pd.read_sql(
                text("""
                    SELECT id AS ID, medicine AS Medicine, dose AS Dose,
                               time AS Time, refill AS Refill
                    FROM medicines
                    ORDER BY id
                """),
                connection,
            )
    except SQLAlchemyError:
        st.error("Unable to read medicines from the database. Check the database file and try again.")
        st.stop()

    if medicines.empty:
        return pd.DataFrame(
            {
                "ID": pd.Series(dtype="int64"),
                "Medicine": pd.Series(dtype="object"),
                "Dose": pd.Series(dtype="object"),
                "Time": pd.Series(dtype="object"),
                "Refill": pd.Series(dtype="int64"),
            }
        )

    medicines["ID"] = medicines["ID"].astype(int)
    medicines["Refill"] = medicines["Refill"].fillna(0).astype(int)
    return medicines


def ensure_dose_occurrences(medicines, scheduled_dates):
    try:
        with engine.begin() as connection:
            for row in medicines.itertuples(index=False):
                for scheduled_date in scheduled_dates:
                    connection.execute(
                        text("""
                            INSERT INTO dose_occurrences
                            (medicine_id, scheduled_date, scheduled_time,
                             medicine_name_snapshot, dose_snapshot, status)
                            SELECT :medicine_id, :scheduled_date, :scheduled_time,
                                   :medicine_name, :dose, 'Pending'
                            WHERE NOT EXISTS (
                                SELECT 1 FROM dose_occurrences
                                WHERE medicine_id = :medicine_id
                                  AND scheduled_date = :scheduled_date
                            )
                        """),
                        {
                            "medicine_id": int(row.ID),
                            "scheduled_date": scheduled_date.isoformat(),
                            "scheduled_time": str(row.Time),
                            "medicine_name": str(row.Medicine),
                            "dose": str(row.Dose),
                        },
                    )
    except SQLAlchemyError as error:
        st.error(f"Could not prepare dated doses in the database: {error}")
        st.stop()


def load_today_doses():
    try:
        with engine.connect() as connection:
            doses = pd.read_sql(
                text("""
                    SELECT occurrence.id AS "Occurrence ID",
                           medicines.id AS ID,
                           medicines.medicine AS Medicine,
                           medicines.dose AS Dose,
                           occurrence.scheduled_time AS Time,
                           occurrence.status = 'Taken' AS Taken,
                           medicines.refill AS Refill
                    FROM dose_occurrences AS occurrence
                    JOIN medicines ON medicines.id = occurrence.medicine_id
                    WHERE occurrence.scheduled_date = :scheduled_date
                    ORDER BY occurrence.scheduled_time, medicines.id
                """),
                connection,
                params={"scheduled_date": date.today().isoformat()},
            )
    except SQLAlchemyError as error:
        st.error(f"Unable to read today's doses from the database: {error}")
        st.stop()

    if not doses.empty:
        doses["Occurrence ID"] = doses["Occurrence ID"].astype(int)
        doses["ID"] = doses["ID"].astype(int)
        doses["Taken"] = doses["Taken"].astype(bool)
        doses["Refill"] = doses["Refill"].fillna(0).astype(int)
    return doses


def load_dose_history():
    try:
        with engine.connect() as connection:
            return pd.read_sql(
                text("""
                    SELECT occurrence.medicine_name_snapshot AS Medicine,
                           occurrence.dose_snapshot AS Dose,
                           occurrence.scheduled_date AS "Scheduled date",
                           occurrence.scheduled_time AS "Scheduled time",
                           occurrence.status AS Status,
                           occurrence.taken_at AS "Taken time"
                    FROM dose_occurrences AS occurrence
                    JOIN medicines ON medicines.id = occurrence.medicine_id
                    ORDER BY occurrence.scheduled_date DESC,
                             occurrence.scheduled_time DESC,
                             occurrence.id DESC
                """),
                connection,
            )
    except SQLAlchemyError as error:
        st.error(f"Unable to read dose history from the database: {error}")
        st.stop()


def duplicate_exists(connection, medicine, dose, medicine_time, exclude_id=None):
    saved_medicines = connection.execute(
        text("SELECT id, medicine, dose, time FROM medicines")
    )
    for saved_id, saved_name, saved_dose, saved_time in saved_medicines:
        if exclude_id is not None and int(saved_id) == int(exclude_id):
            continue
        if (
            str(saved_name).strip().casefold() == medicine.casefold()
            and str(saved_dose).strip().casefold() == dose.casefold()
            and normalized_time(saved_time) == normalized_time(medicine_time)
        ):
            return True
    return False


def medicine_label(medicine_id):
    row = MEDICINES.loc[MEDICINES["ID"] == medicine_id].iloc[0]
    return f"{row['Medicine']} — {row['Dose']} — {row['Time']}"


def medicine_choices_key():
    choices = tuple(
        (int(row.ID), str(row.Medicine), str(row.Dose), str(row.Time))
        for row in MEDICINES.itertuples(index=False)
    )
    return hash(choices)


def get_next_reminder(doses):
    now = datetime.now()
    upcoming = []
    for _, row in doses.iterrows():
        if row["Status"] != "Pending":
            continue
        try:
            scheduled_time = parse_medicine_time(row["Scheduled time"])
            scheduled_date = date.fromisoformat(str(row["Scheduled date"]))
            if scheduled_time is None:
                continue
            scheduled_datetime = datetime.combine(scheduled_date, scheduled_time)
        except (TypeError, ValueError):
            continue
        if scheduled_datetime >= now:
            upcoming.append((scheduled_datetime, row))
    if not upcoming:
        return None
    return min(upcoming, key=lambda item: item[0])[1]


def get_reminder_status(doses, now=None):
    now = (now or datetime.now()).replace(second=0, microsecond=0)
    statuses = {"Due Now": [], "Overdue": [], "Upcoming": []}
    invalid_count = 0

    for _, row in doses.iterrows():
        if row["Status"] != "Pending":
            continue
        try:
            scheduled_time = parse_medicine_time(row["Scheduled time"])
            scheduled_date = date.fromisoformat(str(row["Scheduled date"]))
            if scheduled_time is None:
                invalid_count += 1
                continue
            scheduled_datetime = datetime.combine(scheduled_date, scheduled_time)
        except (TypeError, ValueError):
            invalid_count += 1
            continue

        status = (
            "Upcoming"
            if scheduled_datetime > now
            else "Due Now"
            if scheduled_datetime == now
            else "Overdue"
        )
        statuses[status].append(
            {
                "Medicine": row["Medicine"],
                "Dose": row["Dose"],
                "Scheduled date": row["Scheduled date"],
                "Scheduled time": row["Scheduled time"],
                "_scheduled_datetime": scheduled_datetime,
            }
        )

    for status_rows in statuses.values():
        status_rows.sort(key=lambda row: row["_scheduled_datetime"])
        for row in status_rows:
            del row["_scheduled_datetime"]
    return statuses, invalid_count


today = date.today()
MEDICINES = load_medicines()
ensure_dose_occurrences(MEDICINES, [today, today + timedelta(days=1)])

st.title("Pillcare")
st.caption("Your medicine routine, made easier to follow.")
with st.expander("➕ Add Medicine"):
    with st.form("add_medicine_form"):
        medicine_name = st.text_input("Medicine name")
        dose = st.text_input("Dose", placeholder="Example: 500 mg")
        medicine_time = st.text_input("Time", placeholder="Example: 9:00 AM")
        refill = st.number_input("Supply left (days)", min_value=0, step=1)

        submitted = st.form_submit_button("Add Medicine")

        if submitted:
            medicine, validation_error = validate_medicine(
                medicine_name, dose, medicine_time, int(refill)
            )
            if validation_error:
                st.warning(validation_error)
            else:
                try:
                    was_added = False
                    with engine.begin() as connection:
                        if duplicate_exists(
                            connection,
                            medicine["medicine"],
                            medicine["dose"],
                            medicine["time"],
                        ):
                            st.warning(
                                "This medicine, dose, and time are already saved."
                            )
                        else:
                            connection.execute(
                                text("""
                                    INSERT INTO medicines
                                    (medicine, dose, time, taken, refill)
                                    VALUES (:medicine, :dose, :time, :taken, :refill)
                                """),
                                {
                                    **medicine,
                                    "taken": False,
                                },
                            )
                            was_added = True
                    if was_added:
                        st.success(f"{medicine['medicine']} added successfully!")
                        MEDICINES = load_medicines()
                        ensure_dose_occurrences(
                            MEDICINES, [today, today + timedelta(days=1)]
                        )
                except SQLAlchemyError as error:
                    st.error(f"Could not save the medicine to the database: {error}")

with st.expander("🗑️ Delete Medicine"):
    if MEDICINES.empty:
        st.info("No saved medicines to delete.")
    else:
        medicine_id = st.selectbox(
            "Select a saved medicine",
            MEDICINES["ID"].tolist(),
            format_func=medicine_label,
            key=f"delete_medicine_id_{medicine_choices_key()}",
        )

        if st.button("Delete Selected Medicine"):
            try:
                with engine.begin() as connection:
                    connection.execute(
                        text("DELETE FROM dose_occurrences WHERE medicine_id = :id"),
                        {"id": int(medicine_id)},
                    )
                    connection.execute(
                        text("DELETE FROM medicines WHERE id = :id"),
                        {"id": int(medicine_id)},
                    )
                st.success("Medicine deleted successfully!")
                st.rerun()
            except SQLAlchemyError as error:
                st.error(f"Could not delete the medicine from the database: {error}")

with st.expander("✏️ Edit Medicine"):
    if MEDICINES.empty:
        st.info("No saved medicines to edit.")
    else:
        selected_id = st.selectbox(
            "Select a saved medicine to edit",
            MEDICINES["ID"].tolist(),
            format_func=medicine_label,
            key=f"edit_medicine_id_{medicine_choices_key()}",
        )

        selected_row = MEDICINES.loc[MEDICINES["ID"] == selected_id].iloc[0]
        new_name = st.text_input(
            "Medicine name",
            value=selected_row["Medicine"],
            key=f"edit_name_{selected_id}",
        )
        new_dose = st.text_input(
            "Dose",
            value=selected_row["Dose"],
            key=f"edit_dose_{selected_id}",
        )
        new_time = st.text_input(
            "Time",
            value=selected_row["Time"],
            key=f"edit_time_{selected_id}",
        )
        stored_refill = int(selected_row["Refill"])
        if stored_refill < 0:
            st.warning("This medicine has a negative saved supply value. Enter a valid value to correct it.")
        new_refill = st.number_input(
            "Supply left (days)",
            min_value=0,
            value=max(0, stored_refill),
            step=1,
            key=f"edit_refill_{selected_id}",
        )

        if st.button("Save Changes"):
            medicine, validation_error = validate_medicine(
                new_name, new_dose, new_time, int(new_refill)
            )
            if validation_error:
                st.warning(validation_error)
            else:
                try:
                    was_updated = False
                    with engine.begin() as connection:
                        if duplicate_exists(
                            connection,
                            medicine["medicine"],
                            medicine["dose"],
                            medicine["time"],
                            exclude_id=selected_id,
                        ):
                            st.warning(
                                "Another saved medicine already has this name, dose, and time."
                            )
                        else:
                            connection.execute(
                                text("""
                                    UPDATE medicines
                                    SET medicine = :medicine,
                                        dose = :dose,
                                        time = :time,
                                        refill = :refill
                                    WHERE id = :id
                                """),
                                {**medicine, "id": int(selected_id)},
                            )
                            if normalized_time(selected_row["Time"]) != normalized_time(
                                medicine["time"]
                            ):
                                connection.execute(
                                    text("""
                                        UPDATE dose_occurrences
                                        SET scheduled_time = :scheduled_time
                                        WHERE medicine_id = :id
                                          AND scheduled_date >= :today
                                          AND status = 'Pending'
                                    """),
                                    {
                                        "scheduled_time": medicine["time"],
                                        "id": int(selected_id),
                                        "today": today.isoformat(),
                                    },
                                )
                            was_updated = True
                    if was_updated:
                        st.success("Medicine updated successfully!")
                        MEDICINES = load_medicines()
                        st.rerun()
                except SQLAlchemyError as error:
                    st.error(f"Could not update the medicine in the database: {error}")

with st.sidebar:
    st.subheader("Today")
    st.write(date.today().strftime("%A, %B %d"))
    st.divider()
    st.markdown("**Next reminder**")
    next_reminder = get_next_reminder(load_dose_history())
    if next_reminder is None:
        st.write("No upcoming reminders")
    else:
        st.write(f"{next_reminder['Medicine']} · {next_reminder['Dose']}")
        reminder_date = date.fromisoformat(str(next_reminder["Scheduled date"]))
        reminder_label = (
            "Today" if reminder_date == today else reminder_date.strftime("%A, %B %d")
        )
        st.caption(f"{reminder_label} · {next_reminder['Scheduled time']}")
    st.divider()
    st.caption("Medicine reminders are for tracking only. Confirm medicine information with a healthcare professional.")


@st.fragment(run_every="30s")
def show_medicine_reminder_status():
    st.subheader("Medicine Reminder Status")
    st.caption(
        "Reminders are displayed while Pillcare is open. Pillcare does not provide "
        "guaranteed device notifications when the application is closed."
    )
    statuses, invalid_count = get_reminder_status(load_dose_history())
    stages = ("Due Now", "Overdue", "Upcoming")
    columns = st.columns(len(stages))
    for column, status in zip(columns, stages):
        column.metric(status, len(statuses[status]))

    reminder_rows = []
    for status in stages:
        reminder_rows.extend(
            {"Status": status, **dose} for dose in statuses[status]
        )
    if reminder_rows:
        st.dataframe(
            pd.DataFrame(reminder_rows),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No pending doses to display.")
    if invalid_count:
        st.warning(
            f"{invalid_count} pending dose(s) with an invalid date or time were skipped."
        )


show_medicine_reminder_status()

st.subheader("Today's schedule")
TODAY_DOSES = load_today_doses()
selected = st.data_editor(
    TODAY_DOSES,
    hide_index=True,
    width="stretch",
    disabled=["Occurrence ID", "ID", "Medicine", "Dose", "Time", "Refill"],
    column_config={
        "Occurrence ID": None,
        "ID": None,
        "Taken": st.column_config.CheckboxColumn("Taken", help="Mark a dose as taken"),
        "Refill": st.column_config.NumberColumn("Supply left (days)", format="%d days"),
    },
    key="medicine_schedule",
)

taken_changes = []
for _, row in selected.iterrows():
    previous = TODAY_DOSES.loc[
        TODAY_DOSES["Occurrence ID"] == row["Occurrence ID"], "Taken"
    ]
    if not previous.empty and bool(previous.iloc[0]) != bool(row["Taken"]):
        taken_changes.append((bool(row["Taken"]), int(row["Occurrence ID"])))

if taken_changes:
    try:
        with engine.begin() as connection:
            connection.execute(
                text("""
                    UPDATE dose_occurrences
                    SET status = :status,
                        taken_at = CASE WHEN :taken = 1 THEN CURRENT_TIMESTAMP ELSE NULL END
                    WHERE id = :id AND scheduled_date = :scheduled_date
                """),
                [
                    {
                        "taken": taken,
                        "status": "Taken" if taken else "Pending",
                        "id": occurrence_id,
                        "scheduled_date": today.isoformat(),
                    }
                    for taken, occurrence_id in taken_changes
                ],
            )
        st.rerun()
    except SQLAlchemyError as error:
        st.error(f"Could not save dose status to the database: {error}")

completed = int(selected["Taken"].sum())
col1, col2, col3 = st.columns(3)
col1.metric("Today's doses", len(selected))
col2.metric("Marked taken", f"{completed} / {len(selected)}")
col3.metric("Refill soon", int((MEDICINES["Refill"] <= 7).sum()))

left, right = st.columns([3, 2])
with left:
    st.subheader("Medicine information")
    query = st.text_input("Search your medicine list", placeholder="Try Vitamin D3")
    matches = (
        selected[
            selected["Medicine"].str.contains(
                query, case=False, na=False, regex=False
            )
        ]
        if query
        else selected
    )
    if matches.empty:
        st.info("No matching medicines in today's list.")
    else:
        st.dataframe(
            matches[["Medicine", "Dose", "Time", "Refill"]],
            hide_index=True,
            width="stretch",
        )

with right:
    st.subheader("Dose progress")
    progress = selected["Taken"].mean() if not selected.empty else 0
    st.progress(float(progress), text=f"{int(progress * 100)}% of today's doses")
    if selected.empty:
        st.info("No medicine supply data yet.")
    else:
        fig, ax = plt.subplots(figsize=(5, 2.4))
        sns.barplot(data=selected, x="Medicine", y="Refill", hue="Medicine", legend=False, ax=ax, palette="crest")
        ax.set_ylabel("Days of supply")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

st.subheader("Medicine Information & Reference")
st.caption(
    'Source: [U.S. National Library of Medicine (RxNorm)]'
    '(https://www.nlm.nih.gov/research/umls/rxnorm/). '
    '[RxNav name-search API documentation]'
    '(https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.findRxcuiByString.html).'
)
st.caption(
    "RxNorm is a U.S. medicine terminology/reference source and does not represent "
    "Indian regulatory approval, availability, or prescribing advice."
)
st.caption(
    "Pillcare provides reference information only and does not replace advice from a doctor or pharmacist."
)

with st.form("rxnorm_search_form"):
    rxnorm_query = st.text_input(
        "Medicine name",
        placeholder="Enter a medicine name, such as Paracetamol",
        key="rxnorm_medicine_query",
    )
    rxnorm_submitted = st.form_submit_button("Search RxNorm")

if rxnorm_submitted:
    if not rxnorm_query.strip():
        st.warning("Enter a medicine name to search.")
    else:
        with st.spinner("Searching the RxNorm reference..."):
            rxnorm_status, rxnorm_results = search_rxnorm_medicine(
                rxnorm_query.strip()
            )

        if rxnorm_status == "results":
            st.dataframe(
                pd.DataFrame(rxnorm_results),
                hide_index=True,
                width="stretch",
            )
            if len({tuple(result.keys()) for result in rxnorm_results}) > 1:
                st.caption("Some fields are not provided for every result.")
        elif rxnorm_status == "not_found":
            st.info("No matching medicine was found in RxNorm.")
        elif rxnorm_status == "missing_fields":
            st.warning("RxNorm returned a result without the supported display fields.")
        elif rxnorm_status == "timeout":
            st.warning("The RxNorm request timed out. Please try again later.")
        elif rxnorm_status == "rate_limit":
            st.warning("RxNorm is rate-limiting requests. Please wait and try again.")
        elif rxnorm_status == "invalid_response":
            st.warning("RxNorm returned an invalid or incomplete response.")
        else:
            st.warning(
                "The RxNorm service is unavailable or the request failed. "
                "You can continue using the rest of Pillcare."
            )

with st.expander("Sample demo medicines"):
    st.caption("Examples only; these are not saved and are excluded from your dashboard totals.")
    st.dataframe(DEMO_MEDICINES, hide_index=True, width="stretch")

with st.expander("Dose History"):
    dose_history = load_dose_history()
    if dose_history.empty:
        st.info("No dated doses have been recorded yet.")
    else:
        if dose_history[["Medicine", "Dose"]].isna().any(axis=None):
            st.caption(
                "Medicine name or dose is blank for older records because it was "
                "not stored when those occurrences were created."
            )
        st.dataframe(dose_history, hide_index=True, width="stretch")

with st.expander("Project integration starter"):
    st.write("Connect a trusted medicine-information source and persistent reminder storage here.")
    st.code(
        "# Example integration points\n"
        "engine = create_engine('sqlite:///pillcare.db')\n"
        "response = requests.get('YOUR_TRUSTED_API_URL', timeout=10)\n"
        "vectorizer = TfidfVectorizer()  # for local text search experiments\n"
        "sample_days = np.array([item for item in selected['Refill']])\n"
        "with engine.connect() as connection:\n"
        "    connection.execute(text('SELECT 1'))",
        language="python",
    )
