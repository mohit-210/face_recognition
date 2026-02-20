import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';



import 'package:http/http.dart' as http;

class FaceApiException implements Exception {
  FaceApiException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() => 'FaceApiException(statusCode: $statusCode, message: $message)';
}


class AuthTokens {
  AuthTokens({
    required this.accessToken,
    required this.refreshToken,
    required this.tokenType,
  });

  final String accessToken;
  final String refreshToken;
  final String tokenType;

  factory AuthTokens.fromJson(Map<String, dynamic> json) {
    return AuthTokens(
      accessToken: json['access_token'] as String,
      refreshToken: json['refresh_token'] as String,
      tokenType: (json['token_type'] as String?) ?? 'bearer',
    );
  }
}

class UserRead {
  UserRead({
    required this.id,
    required this.companyId,
    required this.name,
    required this.employeeCode,
    required this.status,
  });

  final int id;
  final int companyId;
  final String name;
  final String employeeCode;
  final String status;

  factory UserRead.fromJson(Map<String, dynamic> json) {
    return UserRead(
      id: json['id'] as int,
      companyId: json['company_id'] as int,
      name: json['name'] as String,
      employeeCode: json['employee_code'] as String,
      status: json['status'] as String,
    );
  }
}

class FaceRegisterResponse {
  FaceRegisterResponse({
    required this.userId,
    required this.embeddingsSaved,
  });

  final int userId;
  final int embeddingsSaved;

  factory FaceRegisterResponse.fromJson(Map<String, dynamic> json) {
    return FaceRegisterResponse(
      userId: json['user_id'] as int,
      embeddingsSaved: json['embeddings_saved'] as int,
    );
  }
}

class FaceVerifyResponse {
  FaceVerifyResponse({
    required this.verified,
    required this.confidence,
    required this.livenessScore,
    required this.reason,
  });

  final bool verified;
  final double confidence;
  final double livenessScore;
  final String reason;

  factory FaceVerifyResponse.fromJson(Map<String, dynamic> json) {
    return FaceVerifyResponse(
      verified: json['verified'] as bool,
      confidence: (json['confidence'] as num).toDouble(),
      livenessScore: (json['liveness_score'] as num).toDouble(),
      reason: json['reason'] as String,
    );
  }
}

class AttendanceRead {
  AttendanceRead({
    required this.id,
    required this.companyId,
    required this.userId,
    required this.attendanceDate,
    required this.firstCheckInAt,
    required this.lastCheckOutAt,
    required this.lastSeenAt,
    required this.verificationCount,
    required this.status,
  });

  final int id;
  final int companyId;
  final int userId;
  final DateTime attendanceDate;
  final DateTime? firstCheckInAt;
  final DateTime? lastCheckOutAt;
  final DateTime? lastSeenAt;
  final int verificationCount;
  final String status;

  factory AttendanceRead.fromJson(Map<String, dynamic> json) {
    DateTime? parseTs(Object? value) => value == null ? null : DateTime.parse(value as String);
    return AttendanceRead(
      id: json['id'] as int,
      companyId: json['company_id'] as int,
      userId: json['user_id'] as int,
      attendanceDate: DateTime.parse(json['attendance_date'] as String),
      firstCheckInAt: parseTs(json['first_check_in_at']),
      lastCheckOutAt: parseTs(json['last_check_out_at']),
      lastSeenAt: parseTs(json['last_seen_at']),
      verificationCount: json['verification_count'] as int,
      status: json['status'] as String,
    );
  }
}

class AttendanceMarkRead {
  AttendanceMarkRead({
    required this.action,
    required this.record,
  });

  final String action;
  final AttendanceRead record;

  factory AttendanceMarkRead.fromJson(Map<String, dynamic> json) {
    return AttendanceMarkRead(
      action: json['action'] as String,
      record: AttendanceRead.fromJson(json['record'] as Map<String, dynamic>),
    );
  }
}

class AttendanceScanResponse {
  AttendanceScanResponse({
    required this.verified,
    required this.userId,
    required this.name,
    required this.confidence,
    required this.livenessScore,
    required this.reason,
    required this.attendance,
    required this.samplesEvaluated,
    required this.samplesVerified,
    required this.consensusRatio,
  });

  final bool verified;
  final int? userId;
  final String? name;
  final double confidence;
  final double livenessScore;
  final String reason;
  final AttendanceMarkRead? attendance;
  final int? samplesEvaluated;
  final int? samplesVerified;
  final double? consensusRatio;

  factory AttendanceScanResponse.fromJson(Map<String, dynamic> json) {
    return AttendanceScanResponse(
      verified: json['verified'] as bool,
      userId: json['user_id'] as int?,
      name: json['name'] as String?,
      confidence: (json['confidence'] as num).toDouble(),
      livenessScore: (json['liveness_score'] as num).toDouble(),
      reason: json['reason'] as String,
      attendance: json['attendance'] == null
          ? null
          : AttendanceMarkRead.fromJson(json['attendance'] as Map<String, dynamic>),
      samplesEvaluated: json['samples_evaluated'] as int?,
      samplesVerified: json['samples_verified'] as int?,
      consensusRatio: (json['consensus_ratio'] as num?)?.toDouble(),
    );
  }
}

class CreateUserWithFaceResult {
  CreateUserWithFaceResult({
    required this.user,
    required this.faceRegistration,
  });

  final UserRead user;
  final FaceRegisterResponse faceRegistration;
}

class FaceApiClient {
  FaceApiClient({
    required this.baseUrl,
    http.Client? httpClient,
  }) : _httpClient = httpClient ?? http.Client();

  final String baseUrl;
  final http.Client _httpClient;

  AuthTokens? _tokens;

  bool get isLoggedIn => _tokens != null;
  String? get accessToken => _tokens?.accessToken;

  Future<AuthTokens> login({
    required int companyId,
    required String employeeCode,
    required String password,
  }) async {
    final response = await _httpClient.post(
      Uri.parse('$baseUrl/api/v1/auth/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'company_id': companyId,
        'employee_code': employeeCode,
        'password': password,
      }),
    );
    final map = _decodeMap(response);
    if (response.statusCode >= 400) {
      throw FaceApiException(_extractDetail(map), statusCode: response.statusCode);
    }
    final tokens = AuthTokens.fromJson(map);
    _tokens = tokens;
    return tokens;
  }

  Future<AuthTokens> refreshToken() async {
    final current = _tokens;
    if (current == null) {
      throw FaceApiException('Not logged in');
    }
    final response = await _httpClient.post(
      Uri.parse('$baseUrl/api/v1/auth/refresh'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'refresh_token': current.refreshToken}),
    );
    final map = _decodeMap(response);
    if (response.statusCode >= 400) {
      throw FaceApiException(_extractDetail(map), statusCode: response.statusCode);
    }
    final refreshed = AuthTokens.fromJson(map);
    _tokens = refreshed;
    return refreshed;
  }

  Future<UserRead> createUser({
    required int companyId,
    required String name,
    required String employeeCode,
    required String password,
  }) async {
    final response = await _sendAuthorized(
      method: 'POST',
      path: '/api/v1/users',
      jsonBody: {
        'company_id': companyId,
        'name': name,
        'employee_code': employeeCode,
        'password': password,
      },
    );
    return UserRead.fromJson(_decodeMap(response));
  }

  Future<FaceRegisterResponse> registerFace({
    required int userId,
    required List<Uint8List> imageBytes,
  }) async {
    if (imageBytes.length < 3) {
      throw FaceApiException('At least 3 photos are required for face registration.');
    }
    final imagesBase64 = imageBytes.map(base64Encode).toList();
    final response = await _sendAuthorized(
      method: 'POST',
      path: '/api/v1/face/register',
      jsonBody: {
        'user_id': userId,
        'images_base64': imagesBase64,
      },
    );
    return FaceRegisterResponse.fromJson(_decodeMap(response));
  }

  Future<CreateUserWithFaceResult> addUserWithPhotoRequired({
    required int companyId,
    required String name,
    required String employeeCode,
    required String password,
    required List<Uint8List> photos,
  }) async {
    final user = await createUser(
      companyId: companyId,
      name: name,
      employeeCode: employeeCode,
      password: password,
    );
    final registration = await registerFace(userId: user.id, imageBytes: photos);
    return CreateUserWithFaceResult(user: user, faceRegistration: registration);
  }

  Future<FaceVerifyResponse> verifyFace({
    required int companyId,
    required int userId,
    required Uint8List imageBytes,
    Uint8List? previousImageBytes,
    String? expectedChallenge,
    String? challengeResponse,
    String? deviceId,
  }) async {
    final response = await _sendAuthorized(
      method: 'POST',
      path: '/api/v1/face/verify',
      jsonBody: {
        'company_id': companyId,
        'user_id': userId,
        'image_base64': base64Encode(imageBytes),
        'previous_image_base64': previousImageBytes == null ? null : base64Encode(previousImageBytes),
        'expected_challenge': expectedChallenge,
        'challenge_response': challengeResponse,
        'device_id': deviceId,
      },
    );
    return FaceVerifyResponse.fromJson(_decodeMap(response));
  }

  Future<AttendanceScanResponse> scanFaceForAttendance({
    required Uint8List imageBytes,
    Uint8List? previousImageBytes,
    String? deviceId,
    bool markAttendance = true,
  }) async {
    final response = await _sendAuthorized(
      method: 'POST',
      path: '/api/v1/attendance/scan-face',
      jsonBody: {
        'image_base64': base64Encode(imageBytes),
        'previous_image_base64': previousImageBytes == null ? null : base64Encode(previousImageBytes),
        'device_id': deviceId,
        'mark_attendance': markAttendance,
      },
    );
    return AttendanceScanResponse.fromJson(_decodeMap(response));
  }

  Future<AttendanceScanResponse> scanFaceBurstForAttendance({
    required List<Uint8List> imageBytesBurst,
    String? deviceId,
    bool markAttendance = true,
    int minVerifiedSamples = 1,
    double minConsensusRatio = 0.60,
  }) async {
    if (imageBytesBurst.length < 2) {
      throw FaceApiException('At least 2 frames are required for burst attendance scan.');
    }
    if (imageBytesBurst.length > 5) {
      throw FaceApiException('A maximum of 5 frames is allowed for burst attendance scan.');
    }
    final response = await _sendAuthorized(
      method: 'POST',
      path: '/api/v1/attendance/scan-face-burst',
      jsonBody: {
        'images_base64': imageBytesBurst.map(base64Encode).toList(growable: false),
        'device_id': deviceId,
        'mark_attendance': markAttendance,
        'min_verified_samples': minVerifiedSamples,
        'min_consensus_ratio': minConsensusRatio,
      },
    );
    return AttendanceScanResponse.fromJson(_decodeMap(response));
  }

  Future<List<AttendanceRead>> listAttendance({
    DateTime? date,
    int? userId,
  }) async {
    final query = <String, String>{};
    if (date != null) {
      query['attendance_date'] = date.toIso8601String().split('T').first;
    }
    if (userId != null) {
      query['user_id'] = userId.toString();
    }
    final uri = Uri.parse('$baseUrl/api/v1/attendance').replace(queryParameters: query.isEmpty ? null : query);
    final response = await _sendAuthorizedRaw(
      method: 'GET',
      uri: uri,
    );
    final decoded = jsonDecode(response.body);
    if (decoded is! List) {
      throw FaceApiException('Unexpected attendance response format.');
    }
    return decoded
        .map((item) => AttendanceRead.fromJson(item as Map<String, dynamic>))
        .toList(growable: false);
  }

  Future<AttendanceMarkRead> manualCheckout({required int userId}) async {
    final response = await _sendAuthorized(
      method: 'POST',
      path: '/api/v1/attendance/users/$userId/checkout',
    );
    return AttendanceMarkRead.fromJson(_decodeMap(response));
  }

  Future<http.Response> _sendAuthorized({
    required String method,
    required String path,
    Map<String, dynamic>? jsonBody,
    bool retryAfterRefresh = true,
  }) async {
    final token = _tokens?.accessToken;
    if (token == null) {
      throw FaceApiException('Not logged in');
    }

    final uri = Uri.parse('$baseUrl$path');
    final response = await _sendAuthorizedRaw(
      method: method,
      uri: uri,
      jsonBody: jsonBody,
    );

    if (response.statusCode == 401 && retryAfterRefresh && _tokens != null) {
      await refreshToken();
      return _sendAuthorized(
        method: method,
        path: path,
        jsonBody: jsonBody,
        retryAfterRefresh: false,
      );
    }

    if (response.statusCode >= 400) {
      final map = _decodeMap(response);
      throw FaceApiException(_extractDetail(map), statusCode: response.statusCode);
    }
    return response;
  }

  Future<http.Response> _sendAuthorizedRaw({
    required String method,
    required Uri uri,
    Map<String, dynamic>? jsonBody,
  }) async {
    final token = _tokens?.accessToken;
    if (token == null) {
      throw FaceApiException('Not logged in');
    }

    final headers = {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer $token',
    };
    final body = jsonBody == null ? null : jsonEncode(jsonBody);

    switch (method.toUpperCase()) {
      case 'POST':
        return _httpClient.post(uri, headers: headers, body: body);
      case 'PUT':
        return _httpClient.put(uri, headers: headers, body: body);
      case 'GET':
        return _httpClient.get(uri, headers: headers);
      case 'DELETE':
        return _httpClient.delete(uri, headers: headers, body: body);
      default:
        throw FaceApiException('Unsupported HTTP method: $method');
    }
  }

  Map<String, dynamic> _decodeMap(http.Response response) {
    if (response.body.trim().isEmpty) return <String, dynamic>{};
    final decoded = jsonDecode(response.body);
    if (decoded is Map<String, dynamic>) {
      return decoded;
    }
    throw FaceApiException('Unexpected API response format.', statusCode: response.statusCode);
  }

  String _extractDetail(Map<String, dynamic> body) {
    final detail = body['detail'];
    if (detail is String && detail.isNotEmpty) return detail;
    if (detail != null) return detail.toString();
    return 'Request failed';
  }

  void logout() {
    _tokens = null;
  }

  void dispose() {
    _httpClient.close();
  }
}

typedef CaptureFrameBytes = Future<Uint8List?> Function();
typedef VerifyResultCallback = void Function(FaceVerifyResponse result);
typedef VerifyErrorCallback = void Function(Object error, StackTrace stackTrace);

class FaceVerificationScanner {
  FaceVerificationScanner({
    required this.apiClient,
    required this.companyId,
    required this.userId,
    required this.captureFrameBytes,
    this.deviceId = 'flutter-app',
    this.pollInterval = const Duration(milliseconds: 900),
    this.onResult,
    this.onError,
  });

  final FaceApiClient apiClient;
  final int companyId;
  final int userId;
  final CaptureFrameBytes captureFrameBytes;
  final String deviceId;
  final Duration pollInterval;
  final VerifyResultCallback? onResult;
  final VerifyErrorCallback? onError;

  Timer? _timer;
  bool _isRequestRunning = false;
  bool _isVerified = false;
  Uint8List? _previousFrame;

  bool get isRunning => _timer != null;
  bool get isVerified => _isVerified;

  void start() {
    if (_timer != null || _isVerified) return;
    _timer = Timer.periodic(pollInterval, (_) => _tick());
  }

  Future<void> _tick() async {
    if (_isRequestRunning || _isVerified) return;
    _isRequestRunning = true;
    try {
      final frame = await captureFrameBytes();
      if (frame == null || frame.isEmpty) {
        _isRequestRunning = false;
        return;
      }

      final result = await apiClient.verifyFace(
        companyId: companyId,
        userId: userId,
        imageBytes: frame,
        previousImageBytes: _previousFrame,
        deviceId: deviceId,
      );
      _previousFrame = frame;
      onResult?.call(result);

      if (result.verified) {
        _isVerified = true;
        stop();
      }
    } catch (e, st) {
      onError?.call(e, st);
    } finally {
      _isRequestRunning = false;
    }
  }

  void stop() {
    _timer?.cancel();
    _timer = null;
  }

  void retry() {
    _isVerified = false;
    _previousFrame = null;
    stop();
    start();
  }

  void dispose() {
    stop();
  }
}
