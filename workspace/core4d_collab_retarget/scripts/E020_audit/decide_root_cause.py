#!/usr/bin/env python3
from audit_common import run_decide_root_cause


if __name__ == "__main__":
    root_csv, summary = run_decide_root_cause()
    print(root_csv)
    print(summary)
