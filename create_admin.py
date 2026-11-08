
from getpass import getpass

from database import init_db, SessionLocal, log_action
from authentication import create_user


def create_admin():

    print("\n" + "=" * 45)
    print("STUDENT CLASSIFICATION SYSTEM")
    print("ADMIN ACCOUNT CREATION")
    print("=" * 45)

    init_db()

    username = input("Enter admin username: ").strip()
    password = getpass("Enter admin password: ")
    confirm = getpass("Confirm admin password: ")

    if not username:
        print("Username cannot be empty.")
        return

    if len(password) < 6:
        print("Password must contain at least 6 characters.")
        return

    if password != confirm:
        print("Passwords do not match.")
        return

    db = SessionLocal()

    try:

        print("\nCreating administrator...")

        success, message = create_user(
            db,
            username,
            password,
            "admin"
        )

        if success:

            log_action(
                db,
                username,
                "CREATE_ADMIN",
                "Administrator account created"
            )

            db.commit()

            print("\nADMIN CREATED SUCCESSFULLY!")
            print("Username:", username)
            print("Role: admin")

        else:

            db.rollback()
            print("\nERROR:", message)

    except Exception as error:

        db.rollback()
        print("\nDATABASE ERROR:", repr(error))

    finally:
        db.close()


if __name__ == "__main__":
    create_admin()
