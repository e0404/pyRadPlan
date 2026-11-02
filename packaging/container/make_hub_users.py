"""Create the users of the JupyterHub of jupyterhub-values.yaml with a password each.

Writes the names and generated passwords to a CSV file for handing them out, and prints
the password hashes for ``hub.config.PasswordListAuthenticator.password_hashes``:

    python make_hub_users.py admin anna ben --output users.csv
    python make_hub_users.py @names.txt --output users.csv   (one name per line)

Only the hashes go to the cluster. Keep the CSV file private and out of the repository.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import secrets
import sys

ITERATIONS = 200_000


def password_hash(password: str, iterations: int = ITERATIONS) -> str:
    """Hash a password in the format PasswordListAuthenticator checks."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def main(argv: list[str] | None = None) -> int:
    """Generate passwords for the given users, write them to a CSV and print the hashes."""
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    parser.add_argument("names", nargs="+", help="user names, or @file with one per line")
    parser.add_argument("--output", type=Path, required=True, help="CSV file of the passwords")
    args = parser.parse_args(argv)

    # JupyterHub compares lower-case names.
    names = [name.strip().lower() for name in args.names if name.strip()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        parser.error(f"names given more than once: {', '.join(duplicates)}")
    if args.output.exists():
        parser.error(f"{args.output} exists; choose another file so no password is lost")

    passwords = {name: secrets.token_urlsafe(12) for name in names}
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user", "password"])
        writer.writerows(passwords.items())

    print("    PasswordListAuthenticator:")
    print("      password_hashes:")
    for name, password in passwords.items():
        # Quoted, so that YAML reads names such as "no" or "on" as text.
        print(f"        {json.dumps(name)}: {password_hash(password)}")
    print(f"Passwords of {len(names)} users written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
