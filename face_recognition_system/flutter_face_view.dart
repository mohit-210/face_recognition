import 'dart:typed_data';
import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'flutter_face_api_client.dart';

class FaceOpsPage extends StatefulWidget {
  const FaceOpsPage({super.key});

  @override
  State<FaceOpsPage> createState() => _FaceOpsPageState();
}

class _FaceOpsPageState extends State<FaceOpsPage> {
  final TextEditingController _baseUrlCtrl = TextEditingController(text: 'http://localhost:8000');
  final TextEditingController _companyIdCtrl = TextEditingController();
  final TextEditingController _adminCodeCtrl = TextEditingController();
  final TextEditingController _adminPasswordCtrl = TextEditingController();

  CameraController? _cameraController;
  List<CameraDescription> _availableCameras = [];
  CameraLensDirection _currentLens = CameraLensDirection.front;
  String _captureQuality = 'medium';

  FaceApiClient? _apiClient;

  bool _initializingCamera = false;
  bool _busy = false;
  bool _loggedIn = false;
  bool _capturing = false;
  bool _monitoringAttendance = false;
  bool _monitorRequestInFlight = false;
  Timer? _attendanceMonitorTimer;
  String? _qualityBeforeMonitor;
  Uint8List? _monitorPreviousFrame;

  String _status = 'Ready';
  String _attendanceStatus = 'No attendance marked yet.';
  String? _activeBaseUrl;
  int? _activeCompanyId;
  AttendanceMarkRead? _lastMarked;
  AttendanceScanResponse? _pendingAttendanceCandidate;
  static const int _attendanceBurstFrames = 2;
  static const Duration _attendanceBurstGap = Duration(milliseconds: 110);
  static const String _monitorDeviceId = 'flutter-attendance-monitor';
  static const String _burstStrictDeviceId = 'flutter-attendance-burst-strict';
  static const String _burstRetryDeviceId = 'flutter-attendance-burst-retry';

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
        _resolutionPresetForQuality(_captureQuality),
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
    _attendanceMonitorTimer?.cancel();
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

  ResolutionPreset _resolutionPresetForQuality(String quality) {
    switch (quality) {
      case 'low':
        return ResolutionPreset.low;
      case 'high':
        return ResolutionPreset.high;
      default:
        return ResolutionPreset.medium;
    }
  }

  Future<void> _setCaptureQuality(String quality) async {
    if (_captureQuality == quality) return;
    setState(() {
      _captureQuality = quality;
      _status = 'Applying camera quality: $quality...';
    });
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

  Future<List<Uint8List>> _captureBurstFrames({
    int frameCount = _attendanceBurstFrames,
    Duration frameGap = _attendanceBurstGap,
  }) async {
    final frames = <Uint8List>[];
    for (var i = 0; i < frameCount; i++) {
      final frame = await _captureFrameBytes();
      if (frame == null || frame.isEmpty) {
        continue;
      }
      frames.add(frame);
      if (i != frameCount - 1) {
        await Future<void>.delayed(frameGap);
      }
    }
    return frames;
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
        _pendingAttendanceCandidate = null;
        _activeBaseUrl = client.baseUrl;
        _activeCompanyId = companyId;
        _status = 'Login successful | url=${client.baseUrl} | company_id=$companyId';
      });
    } catch (e) {
      setState(() => _status = 'Login failed: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  String _timingSummary(Map<String, double>? timings) {
    if (timings == null || timings.isEmpty) return '';
    final total = timings['total_ms'];
    final detect = timings['detect_ms'];
    final live = timings['liveness_ms'];
    final embed = timings['embed_ms'];
    return ' | t=${(total ?? 0).toStringAsFixed(0)}ms'
        ' d=${(detect ?? 0).toStringAsFixed(0)}'
        ' l=${(live ?? 0).toStringAsFixed(0)}'
        ' e=${(embed ?? 0).toStringAsFixed(0)}';
  }

  String _modelSummary(String? modelUsed) {
    if (modelUsed == null || modelUsed.trim().isEmpty) return '';
    return ' | model=$modelUsed';
  }

  String _thresholdSummary(double? threshold) {
    if (threshold == null) return '';
    return ' | thr=${threshold.toStringAsFixed(2)}';
  }

  Future<void> _attendanceMonitorTick() async {
    final client = _apiClient;
    if (!_monitoringAttendance || _busy || !_loggedIn || client == null || _monitorRequestInFlight) return;

    _monitorRequestInFlight = true;
    try {
      final frame = await _captureFrameBytes();
      if (frame == null || frame.isEmpty) return;

      final previous = _monitorPreviousFrame;
      _monitorPreviousFrame = frame;
      if (previous == null || previous.isEmpty) {
        if (mounted) {
          setState(() {
            _attendanceStatus = 'Monitoring live threshold... hold position.';
          });
        }
        return;
      }

      final result = await client.scanFaceForAttendance(
        imageBytes: frame,
        previousImageBytes: previous,
        deviceId: _monitorDeviceId,
        markAttendance: false,
        requireLiveMotion: true,
        fastMode: true,
        debugTiming: true,
      );

      final thr = result.livenessThresholdUsed ?? 0.0;
      final pass = result.livenessThresholdUsed == null
          ? (result.livenessScore >= 0.0)
          : (result.livenessScore >= thr);

      if (mounted) {
        setState(() {
          _attendanceStatus =
              'Monitor: live=${result.livenessScore.toStringAsFixed(3)}'
              '${_thresholdSummary(result.livenessThresholdUsed)}'
              ' | pass=${pass ? "YES" : "NO"}'
              ' | conf=${result.confidence.toStringAsFixed(3)}'
              '${_modelSummary(result.modelUsed)}'
              '${_timingSummary(result.debugTimings)}'
              ' | ${result.reason}';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _attendanceStatus = 'Monitor error: ${_readableError(e)}';
        });
      }
    } finally {
      _monitorRequestInFlight = false;
    }
  }

  Future<void> _setAttendanceMonitoring(bool enabled) async {
    if (enabled == _monitoringAttendance) return;
    if (enabled) {
      _qualityBeforeMonitor = _captureQuality;
      if (_captureQuality == 'high') {
        // Keep monitor responsive while preserving enough detail for liveness.
        await _setCaptureQuality('medium');
      }
      _monitoringAttendance = true;
      _monitorPreviousFrame = null;
      _attendanceMonitorTimer?.cancel();
      _attendanceMonitorTimer = Timer.periodic(
        const Duration(milliseconds: 500),
        (_) => _attendanceMonitorTick(),
      );
      setState(() {
        _attendanceStatus = 'Live threshold monitoring started.';
      });
      return;
    }

    _monitoringAttendance = false;
    _attendanceMonitorTimer?.cancel();
    _attendanceMonitorTimer = null;
    _monitorPreviousFrame = null;
    final restoreQuality = _qualityBeforeMonitor;
    _qualityBeforeMonitor = null;
    if (restoreQuality != null && restoreQuality != _captureQuality) {
      await _setCaptureQuality(restoreQuality);
    }
    setState(() {
      _attendanceStatus = 'Live threshold monitoring stopped.';
    });
  }

  Future<void> _scanFaceForAttendance() async {
    final client = _apiClient;
    if (_busy || !_loggedIn || client == null) return;

    final frames = await _captureBurstFrames(frameCount: 2);
    if (frames.length < 2) {
      setState(() => _attendanceStatus = 'Need 2 live frames. Hold still and retry.');
      return;
    }

    setState(() {
      _busy = true;
      _attendanceStatus = 'Scanning face (no mark yet)...';
    });

    try {
      final burst = await client.scanFaceBurstForAttendance(
        imageBytesBurst: frames,
        deviceId: _monitorDeviceId,
        markAttendance: false,
        requireLiveMotion: true,
        minVerifiedSamples: 1,
        minConsensusRatio: 0.50,
        fastMode: true,
        debugTiming: true,
      );
      if (burst.verified && burst.userId != null && burst.name != null) {
        setState(() {
          _pendingAttendanceCandidate = burst;
          _attendanceStatus =
              'Verified: ${burst.name} (not marked yet)'
              ' | conf=${burst.confidence.toStringAsFixed(3)}'
              ' | live=${burst.livenessScore.toStringAsFixed(3)}'
              '${_thresholdSummary(burst.livenessThresholdUsed)}'
              '${_modelSummary(burst.modelUsed)}'
              '${_timingSummary(burst.debugTimings)}';
        });
        return;
      }

      setState(() {
        _pendingAttendanceCandidate = null;
        _attendanceStatus = '${burst.reason}${_modelSummary(burst.modelUsed)}${_timingSummary(burst.debugTimings)}';
      });
    } catch (e) {
      setState(() {
        _pendingAttendanceCandidate = null;
        _attendanceStatus = 'Attendance scan failed: ${_readableError(e)}';
      });
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _markVerifiedAttendance() async {
    final client = _apiClient;
    final pending = _pendingAttendanceCandidate;
    if (_busy || !_loggedIn || client == null || pending == null || pending.userId == null || pending.name == null) {
      return;
    }

    // Old strict mode path kept (disabled):
    // if (_captureQuality != 'high') {
    //   await _setCaptureQuality('high');
    // }
    // final frames = await _captureBurstFrames(frameCount: 2);
    // if (frames.length < 2) {
    //   setState(() => _attendanceStatus = 'Need 2 live frames to mark attendance. Hold still and retry.');
    //   return;
    // }

    setState(() {
      _busy = true;
      _attendanceStatus = 'Marking attendance for ${pending.name}...';
    });

    try {
      final marked = await client.markVerifiedAttendance(userId: pending.userId!);
      setState(() {
        _lastMarked = marked;
        _pendingAttendanceCandidate = null;
        _attendanceStatus =
            'Marked: ${pending.name} | ${marked.action.replaceAll('_', ' ')}'
            ' | status=${marked.record.status}'
            ' | from pre-scan'
            ' | conf=${pending.confidence.toStringAsFixed(3)}'
            ' | live=${pending.livenessScore.toStringAsFixed(3)}'
            '${_thresholdSummary(pending.livenessThresholdUsed)}'
            '${_modelSummary(pending.modelUsed)}'
            '${_timingSummary(pending.debugTimings)}';
      });

      // Old strict re-check before marking (kept intentionally as requested):
      // final burst = await client.scanFaceBurstForAttendance(
      //   imageBytesBurst: frames,
      //   deviceId: _burstStrictDeviceId,
      //   markAttendance: true,
      //   minVerifiedSamples: 2,
      //   minConsensusRatio: 0.67,
      //   fastMode: true,
      //   debugTiming: true,
      // );
      // if (burst.verified && burst.userId != null && burst.name != null && burst.attendance != null) {
      //   final marked = burst.attendance!;
      //   setState(() {
      //     _lastMarked = marked;
      //     _pendingAttendanceCandidate = null;
      //     _attendanceStatus =
      //         'Marked: ${burst.name} | ${marked.action.replaceAll('_', ' ')}'
      //         ' | status=${marked.record.status}'
      //         ' | conf=${burst.confidence.toStringAsFixed(3)}'
      //         ' | live=${burst.livenessScore.toStringAsFixed(3)}'
      //         ' | consensus=${(100 * (burst.consensusRatio ?? 0)).toStringAsFixed(0)}%'
      //         '${_thresholdSummary(burst.livenessThresholdUsed)}'
      //         '${_modelSummary(burst.modelUsed)}'
      //         '${_timingSummary(burst.debugTimings)}';
      //   });
      //   return;
      // }
      //
      // setState(() {
      //   _pendingAttendanceCandidate = null;
      //   _attendanceStatus =
      //       'Mark failed: ${burst.reason}${_modelSummary(burst.modelUsed)}${_timingSummary(burst.debugTimings)}. '
      //       'Please scan again.';
      // });
    } catch (e) {
      setState(() => _attendanceStatus = 'Attendance mark failed: ${_readableError(e)}');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _readableError(Object error) {
    if (error is FaceApiException) return error.message;
    return error.toString();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;
    final cameraReady = controller != null && controller.value.isInitialized;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Face App Integration'),
        actions: [
          PopupMenuButton<String>(
            onSelected: (value) => _setCaptureQuality(value),
            itemBuilder: (context) => const [
              PopupMenuItem(value: 'low', child: Text('Quality: Low')),
              PopupMenuItem(value: 'medium', child: Text('Quality: Medium')),
              PopupMenuItem(value: 'high', child: Text('Quality: High')),
            ],
            icon: const Icon(Icons.high_quality),
          ),
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
                    child: Text(
                      'Status: $_status | Camera quality: $_captureQuality'
                      ' | url=${_activeBaseUrl ?? "-"} | company_id=${_activeCompanyId ?? "-"}',
                    ),
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
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              'Using IDs -> monitor=$_monitorDeviceId, strict=$_burstStrictDeviceId, retry=$_burstRetryDeviceId',
              style: const TextStyle(fontSize: 12),
            ),
          ),
          const SizedBox(height: 6),
          ElevatedButton(
            onPressed: _busy ? null : _login,
            child: Text(_loggedIn ? 'Logged In' : 'Login'),
          ),
        ],
      ),
    );
  }

  Widget _buildAttendanceSection(bool cameraReady) {
    final pending = _pendingAttendanceCandidate;
    final canMarkPending = pending != null && pending.userId != null && pending.name != null;
    return _sectionCard(
      title: '2) Attendance Two-Step Flow',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: _busy || !_loggedIn || !cameraReady ? null : _scanFaceForAttendance,
                  child: const Text('Scan Face'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: _busy || !_loggedIn || !cameraReady || !canMarkPending ? null : _markVerifiedAttendance,
                  child: Text(
                    canMarkPending
                        ? 'Mark Attendance (${pending!.name})'
                        : 'Mark Attendance',
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: _busy || !_loggedIn || !cameraReady || _monitoringAttendance
                      ? null
                      : () async => _setAttendanceMonitoring(true),
                  child: const Text('Start Live Monitor'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton(
                  onPressed: _monitoringAttendance ? () async => _setAttendanceMonitoring(false) : null,
                  child: const Text('Stop Live Monitor'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(_attendanceStatus),
          const SizedBox(height: 10),
          if (_lastMarked != null) Text('Last Mark: ${_lastMarked!.action} -> ${_lastMarked!.record.status}'),
          const SizedBox(height: 10),
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
