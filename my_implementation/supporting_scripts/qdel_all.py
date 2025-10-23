#!/usr/bin/env python3
import subprocess
import re

USER = "xkaska01"

def get_my_job_ids(user: str):
    try:
        # zavoláme qstat -u <user>
        result = subprocess.run(["qstat", "-u", user],
                                capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Chyba při volání qstat:", e)
        return []

    lines = result.stdout.strip().splitlines()
    job_ids = []

    # regex na zachycení job ID (např. 13823469.pbs-m1.metacentrum.cz)
    job_re = re.compile(r"^(\d+)\.pbs-[\w\.-]+")
    

    for line in lines:
        m = job_re.match(line)
        if m:
            print(m.group()) if line[39:44] == "STDIN" else job_ids.append(m.group(1))  # vypíše celý řádek pro kontrolu
            
    return job_ids

def cancel_jobs(job_ids):
    for job_id in job_ids:
        try:
            print(f"Ruším job {job_id} ...")
            subprocess.run(["qdel", job_id], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Nepodařilo se zrušit job {job_id}: {e}")


if __name__ == "__main__":
    job_ids = get_my_job_ids(USER)
    if not job_ids:
        print("Žádné joby nenalezeny.")
    else:
        print(f"Nalezeno {len(job_ids)} jobů: {' '.join(job_ids)}")
        cancel_jobs(job_ids)
        print("✅ Hotovo.")
