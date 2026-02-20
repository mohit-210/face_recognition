import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'flutter_face_api_client.dart';

class FaceOpsPage extends StatefulWidget {
  const FaceOpsPage({super.key});

  @override
  State<FaceOpsPage> createState() => _FaceOpsPageState();
}

class _FaceOpsPageState extends State<FaceOpsPage> {
  final TextEditingController _baseUrlCtrl = TextEditingController(text: 'http://');
  final TextEditingController _companyIdCtrl = TextEditingController();
  final TextEditingController _adminCodeCtrl = TextEditingController();
  final TextEditingController _adminPasswordCtrl = TextEditingController();

  CameraController? _cameraController;
  List<CameraDescription> _availableCameras = [];
  CameraLensDirection _currentLens = CameraLensDirection.front;

  FaceApiClient? _apiClient;

  bool _initializingCamera = false;
  bool _busy = false;
  bool _loggedIn = false;
  bool _capturing = false;

  String _status = 'Ready';
  String _attendanceStatus = 'No face matched yet.';
  String _approvalLabel = 'Approve Mark In';
  AttendanceScanResponse? _pendingMatch;
  AttendanceMarkRead? _lastMarked;
  Uint8List? _previousAttendanceFrame;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    setState(() {
      _initializingCamera = true;
      _status = 'Initializing camera...';
    });

    try {
      _availableCameras = await availableCameras();

      if (_availableCameras.isEmpty) {
        setState(() {
          _status = 'No camera found on device.';
          _initializingCamera = false;
        });
        return;
      }

      final selected = _availableCameras.firstWhere(
            (c) => c.lensDirection == _currentLens,
        orElse: () => _availableCameras.first,
      );

      final controller = CameraController(
        selected,
        ResolutionPreset.low,
        enableAudio: false,
      );

      await controller.initialize();

      if (!mounted) return;

      setState(() {
        _cameraController = controller;
        _initializingCamera = false;
        _status = 'Camera ready';
      });
    } catch (e) {
      setState(() {
        _status = 'Camera init failed: $e';
        _initializingCamera = false;
      });
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _apiClient?.dispose();
    _baseUrlCtrl.dispose();
    _companyIdCtrl.dispose();
    _adminCodeCtrl.dispose();
    _adminPasswordCtrl.dispose();
    super.dispose();
  }

  Future<void> _switchCamera() async {
    if (_availableCameras.length < 2) return;

    _currentLens = _currentLens == CameraLensDirection.front
        ? CameraLensDirection.back
        : CameraLensDirection.front;

    await _cameraController?.dispose();
    await _initCamera();
  }

  Future<Uint8List?> _captureFrameBytes() async {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) return null;
    if (_capturing) return null;

    _capturing = true;

    try {
      final file = await controller.takePicture();
      return await file.readAsBytes();
    } catch (_) {
      return null;
    } finally {
      _capturing = false;
    }
  }

  Future<void> _login() async {
    if (_busy) return;

    final companyId = int.tryParse(_companyIdCtrl.text.trim());
    if (companyId == null) {
      setState(() => _status = 'Company ID must be a valid number.');
      return;
    }

    setState(() {
      _busy = true;
      _status = 'Logging in...';
    });

    try {
      final client = FaceApiClient(baseUrl: _baseUrlCtrl.text.trim());
      await client.login(
        companyId: companyId,
        employeeCode: _adminCodeCtrl.text.trim(),
        password: _adminPasswordCtrl.text,
      );

      setState(() {
        _apiClient?.dispose();
        _apiClient = client;
        _loggedIn = true;
        _status = 'Login successful';
      });
    } catch (e) {
      setState(() => _status = 'Login failed: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _scanFaceForAttendance() async {
    if (_busy || !_loggedIn || _apiClient == null) return;

    final firstFrame = await _captureFrameBytes();
    if (firstFrame == null || firstFrame.isEmpty) {
      setState(() => _attendanceStatus = 'Unable to capture frame.');
      return;
    }

    setState(() {
      _busy = true;
      _attendanceStatus = 'Matching face in company...';
    });

    try {
      Uint8List currentFrame = firstFrame;
      Uint8List? previousFrame = _previousAttendanceFrame ?? firstFrame;
      if (_previousAttendanceFrame == null) {
        await Future<void>.delayed(const Duration(milliseconds: 120));
        final secondFrame = await _captureFrameBytes();
        if (secondFrame == null || secondFrame.isEmpty) {
          setState(() => _attendanceStatus = 'Unable to capture second live frame.');
          return;
        }
        previousFrame = firstFrame;
        currentFrame = secondFrame;
      }

      final result = await _apiClient!.scanFaceForAttendance(
        imageBytes: currentFrame,
        previousImageBytes: previousFrame,
        deviceId: 'flutter-attendance',
        markAttendance: false,
      );
      _previousAttendanceFrame = currentFrame;

      if (!result.verified || result.userId == null || result.name == null) {
        setState(() {
          _pendingMatch = null;
          _approvalLabel = 'Approve Mark In';
          _attendanceStatus = result.reason;
        });
        return;
      }

      final records = await _apiClient!.listAttendance(
        date: DateTime.now().toUtc(),
        userId: result.userId,
      );
      final currentStatus = records.isNotEmpty ? records.first.status.toLowerCase() : 'out';
      final markOutNext = currentStatus == 'in' || currentStatus == 'present';

      setState(() {
        _pendingMatch = result;
        _approvalLabel = markOutNext ? 'Approve Mark Out' : 'Approve Mark In';
        _attendanceStatus =
            'Matched: ${result.name} (id=${result.userId}) | conf=${result.confidence.toStringAsFixed(3)}';
      });
    } catch (e) {
      setState(() {
        _pendingMatch = null;
        _attendanceStatus = 'Attendance scan failed: $e';
      });
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _approveAttendance() async {
    final client = _apiClient;
    final match = _pendingMatch;
    if (_busy || !_loggedIn || client == null || match == null || match.userId == null) return;

    setState(() {
      _busy = true;
      _attendanceStatus = 'Verifying live face before marking...';
    });

    try {
      final firstFrame = await _captureFrameBytes();
      if (firstFrame == null || firstFrame.isEmpty) {
        setState(() => _attendanceStatus = 'Unable to capture frame for approval.');
        return;
      }

      await Future<void>.delayed(const Duration(milliseconds: 120));
      final secondFrame = await _captureFrameBytes();
      if (secondFrame == null || secondFrame.isEmpty) {
        setState(() => _attendanceStatus = 'Unable to capture second frame for approval.');
        return;
      }

      final approveScan = await client.scanFaceForAttendance(
        imageBytes: secondFrame,
        previousImageBytes: firstFrame,
        deviceId: 'flutter-attendance-approve',
        markAttendance: true,
      );

      if (
          !approveScan.verified ||
          approveScan.userId == null ||
          approveScan.userId != match.userId ||
          approveScan.attendance == null) {
        setState(() {
          _attendanceStatus = approveScan.reason;
        });
        return;
      }

      final marked = approveScan.attendance!;
      final action = marked.action.replaceAll('_', ' ');
      final status = marked.record.status;
      setState(() {
        _lastMarked = marked;
        _pendingMatch = null;
        _previousAttendanceFrame = secondFrame;
        _approvalLabel = 'Approve Mark In';
        _attendanceStatus = 'Marked: ${match.name} | $action | status=$status';
      });
    } catch (e) {
      setState(() => _attendanceStatus = 'Approve failed: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;
    final cameraReady = controller != null && controller.value.isInitialized;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Face App Integration'),
        actions: [
          if (cameraReady)
            IconButton(
              icon: const Icon(Icons.cameraswitch),
              onPressed: _switchCamera,
            ),
        ],
      ),
      body: Column(
        children: [
          if (cameraReady)
            AspectRatio(
              aspectRatio: controller.value.aspectRatio,
              child: CameraPreview(controller),
            )
          else
            Container(
              height: 220,
              alignment: Alignment.center,
              color: Colors.black12,
              child: Text(
                _initializingCamera
                    ? 'Loading camera...'
                    : 'Camera not available',
              ),
            ),

          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  _buildLoginSection(),
                  const SizedBox(height: 12),
                  _buildAttendanceSection(cameraReady),
                  const SizedBox(height: 12),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text('Status: $_status'),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLoginSection() {
    return _sectionCard(
      title: '1) Login',
      child: Column(
        children: [
          _input(_baseUrlCtrl, 'Base URL'),
          _input(_companyIdCtrl, 'Company ID', type: TextInputType.number),
          _input(_adminCodeCtrl, 'Admin Employee Code'),
          _input(_adminPasswordCtrl, 'Admin Password', obscure: true),
          const SizedBox(height: 10),
          ElevatedButton(
            onPressed: _busy ? null : _login,
            child: Text(_loggedIn ? 'Logged In' : 'Login'),
          ),
        ],
      ),
    );
  }

  Widget _buildAttendanceSection(bool cameraReady) {
    final match = _pendingMatch;
    final canApprove = match != null && match.userId != null && !_busy && _loggedIn;

    return _sectionCard(
      title: '2) Attendance Approval Flow',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: _busy || !_loggedIn || !cameraReady ? null : _scanFaceForAttendance,
                  child: const Text('Scan Face (Match Only)'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(_attendanceStatus),
          const SizedBox(height: 10),
          if (match != null) Text('Matched User: ${match.name} (id=${match.userId})'),
          if (_lastMarked != null) Text('Last Mark: ${_lastMarked!.action} -> ${_lastMarked!.record.status}'),
          const SizedBox(height: 10),
          ElevatedButton(
            onPressed: canApprove ? _approveAttendance : null,
            child: Text(_approvalLabel),
          ),
          OutlinedButton(
            onPressed: _busy
                ? null
                : () => setState(() {
                      _pendingMatch = null;
                      _previousAttendanceFrame = null;
                      _approvalLabel = 'Approve Mark In';
                      _attendanceStatus = 'Pending match cleared.';
                    }),
            child: const Text('Clear Pending Match'),
          ),
        ],
      ),
    );
  }

  Widget _input(
    TextEditingController ctrl,
    String label, {
    bool obscure = false,
    TextInputType type = TextInputType.text,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: TextField(
        controller: ctrl,
        obscureText: obscure,
        keyboardType: type,
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
        ),
      ),
    );
  }

  Widget _sectionCard({required String title, required Widget child}) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: Text(title,
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            ),
            const SizedBox(height: 12),
            child,
          ],
        ),
      ),
    );
  }
}
