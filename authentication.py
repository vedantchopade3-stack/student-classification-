
import hashlib
import hmac
import os

from database import User


# PASSWORD HASHING
def hash_password(password):

    salt = os.urandom(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        200000
    )

    return salt.hex(), password_hash.hex()


def verify_password(password, stored_hash, stored_salt):

    try:

        salt = bytes.fromhex(stored_salt)

        calculated_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            200000
        )

        return hmac.compare_digest(
            calculated_hash.hex(),
            stored_hash
        )

    except (ValueError, TypeError, AttributeError):
        return False


# CREATE USER
def create_user(db, username, password, role="faculty"):

    username = username.strip()

    if not username or not password:
        return False, "Username and password are required."

    if len(password) < 6:
        return False, "Password must have at least 6 characters."

    if role not in ["admin", "faculty"]:
        return False, "Invalid user role."

    existing = db.query(User).filter(
        User.username == username
    ).first()

    if existing:
        return False, "Username already exists."

    salt, password_hash = hash_password(password)

    user = User(
        username=username,
        password_hash=password_hash,
        salt=salt,
        role=role
    )

    db.add(user)
    db.flush()

    return True, "User created successfully."


# LOGIN
def login(db, username, password):

    user = db.query(User).filter(
        User.username == username.strip()
    ).first()

    if user is None:
        return None

    if verify_password(
        password,
        user.password_hash,
        user.salt
    ):
        return user

    return None
