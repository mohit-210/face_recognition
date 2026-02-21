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
  String _captureQuality = 'medium';

  FaceApiClient? _apiClient;

  bool _initializingCamera = false;
  bool _busy = false;
  bool _loggedIn = false;
  bool _capturing = false;

  String _status = 'Ready';
  String _attendanceStatus = 'No attendance marked yet.';
  AttendanceMarkRead? _lastMarked;
  static const int _attendanceBurstFrames = 2;
  static const Duration _attendanceBurstGap = Duration(milliseconds: 70);

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
        _status = 'Login successful';
      });
    } catch (e) {
      setState(() => _status = 'Login failed: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  bool _shouldRetryWithBurst(AttendanceScanResponse result) {
    if (result.verified) return false;
    final reason = result.reason.toLowerCase();
    if (reason.contains('multiple faces')) return false;
    return reason.contains('no face') ||
        reason.contains('mismatch') ||
        reason.contains('liveness') ||
        reason.contains('embedding') ||
        result.confidence >= 0.35;
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

  Future<void> _scanAndMarkAttendance() async {
    final client = _apiClient;
    if (_busy || !_loggedIn || client == null) return;

    final frames = await _captureBurstFrames(frameCount: 2);
    if (frames.length < 2) {
      setState(() => _attendanceStatus = 'Need 2 live frames. Hold still and retry.');
      return;
    }

    setState(() {
      _busy = true;
      _attendanceStatus = 'Checking live face and marking attendance...';
    });

    try {
      final burst = await client.scanFaceBurstForAttendance(
        imageBytesBurst: frames,
        deviceId: 'flutter-attendance-burst-strict',
        markAttendance: true,
        minVerifiedSamples: 2,
        minConsensusRatio: 0.67,
        fastMode: false,
        debugTiming: true,
      );
      if (burst.verified && burst.userId != null && burst.name != null && burst.attendance != null) {
        final marked = burst.attendance!;
        setState(() {
          _lastMarked = marked;
          _attendanceStatus =
              'Marked: ${burst.name} | ${marked.action.replaceAll('_', ' ')}'
              ' | status=${marked.record.status}'
              ' | conf=${burst.confidence.toStringAsFixed(3)}'
              ' | live=${burst.livenessScore.toStringAsFixed(3)}'
              ' | consensus=${(100 * (burst.consensusRatio ?? 0)).toStringAsFixed(0)}%'
              '${_thresholdSummary(burst.livenessThresholdUsed)}'
              '${_modelSummary(burst.modelUsed)}'
              '${_timingSummary(burst.debugTimings)}';
        });
        return;
      }

      if (!_shouldRetryWithBurst(burst)) {
        setState(() {
          _attendanceStatus = '${burst.reason}${_modelSummary(burst.modelUsed)}${_timingSummary(burst.debugTimings)}';
        });
        return;
      }

      setState(() => _attendanceStatus = 'Retrying with stronger live check...');
      final retryFrames = await _captureBurstFrames(frameCount: 3);
      if (retryFrames.length < 2) {
        setState(() => _attendanceStatus = 'Retry failed: unable to capture enough live frames.');
        return;
      }

      final retry = await client.scanFaceBurstForAttendance(
        imageBytesBurst: retryFrames,
        deviceId: 'flutter-attendance-burst-retry',
        markAttendance: true,
        minVerifiedSamples: 2,
        minConsensusRatio: 0.67,
        fastMode: false,
        debugTiming: true,
      );
      if (!retry.verified || retry.userId == null || retry.name == null || retry.attendance == null) {
        setState(() {
          _attendanceStatus = '${retry.reason}${_modelSummary(retry.modelUsed)}${_timingSummary(retry.debugTimings)}';
        });
        return;
      }

      final marked = retry.attendance!;
      setState(() {
        _lastMarked = marked;
        _attendanceStatus =
            'Marked (retry): ${retry.name} | ${marked.action.replaceAll('_', ' ')}'
            ' | status=${marked.record.status}'
            ' | conf=${retry.confidence.toStringAsFixed(3)}'
            ' | live=${retry.livenessScore.toStringAsFixed(3)}'
            ' | consensus=${(100 * (retry.consensusRatio ?? 0)).toStringAsFixed(0)}%'
            '${_thresholdSummary(retry.livenessThresholdUsed)}'
            '${_modelSummary(retry.modelUsed)}'
            ' | samples=${retry.samplesVerified ?? 0}/${retry.samplesEvaluated ?? 0}'
            '${_timingSummary(retry.debugTimings)}';
      });
    } catch (e) {
      setState(() => _attendanceStatus = 'Attendance failed: ${_readableError(e)}');
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
                    child: Text('Status: $_status | Camera quality: $_captureQuality'),
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
    return _sectionCard(
      title: '2) Attendance Fast Flow',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: _busy || !_loggedIn || !cameraReady ? null : _scanAndMarkAttendance,
                  child: const Text('Scan & Mark Attendance'),
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
