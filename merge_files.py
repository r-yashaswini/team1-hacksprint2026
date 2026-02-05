import pandas as pd
import json
df1 = pd.read_csv("employee_master.csv")
df2 = pd.read_excel("leave_intelligence.xlsx")
with open("attendance_logs_detailed.json", "r") as f:
    data = json.load(f)
attendance_rows = []
for emp_id, details in data.items():
    for record in details["records"]:
        record["emp_id"] = emp_id
        attendance_rows.append(record)
df3 = pd.DataFrame(attendance_rows)
merged_df = df1.merge(df2, on="emp_id", how="inner") \
                  .merge(df3, on="emp_id", how="inner")
merged_json = merged_df.to_dict(orient="records")
with open("merged.json", "w") as f:
    json.dump(merged_json, f, indent=2)
print("Merged files")
