from app.database.session import Base

# Import models for metadata registration.
from app.models.company import Company  # noqa: F401
from app.models.user import User  # noqa: F401
from app.models.face_profile import FaceProfile  # noqa: F401
from app.models.verification_log import VerificationLog  # noqa: F401
from app.models.attendance_record import AttendanceRecord  # noqa: F401
