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
  final TextEditingController _baseUrlCtrl = TextEditingController(text: 'http://10.0.2.2:8000');
  final TextEditingController _companyIdCtrl = TextEditingController();
  final TextEditingController _adminCodeCtrl = TextEditingController();
  final TextEditingController _adminPasswordCtrl = TextEditingController();

  final TextEditingController _newNameCtrl = TextEditingController();
  final TextEditingController _newEmployeeCodeCtrl = TextEditingController();
  final TextEditingController _newPasswordCtrl = TextEditingController();

  final TextEditingController _verifyUserIdCtrl = TextEditingController();

  CameraController? _cameraController;
  FaceApiClient? _apiClient;
  FaceVerificationScanner? _scanner;

  final List<Uint8List> _enrollPhotos = <Uint8List>[];

  bool _initializingCamera = false;
  bool _busy = false;
  bool _loggedIn = false;
  bool _capturing = false;
  bool _scanning = false;
  bool _verifiedLocked = false;

  String _status = 'Ready';
  String _scanStatus = 'Not started';

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
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _status = 'No camera found on device.';
          _initializingCamera = false;
        });
        return;
      }

      final CameraDescription selected = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        selected,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await controller.initialize();

      if (!mounted) return;
      setState(() {
        _cameraController = controller;
        _status = 'Camera ready';
        _initializingCamera = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Camera init failed: $e';
        _initializingCamera = false;
      });
    }
  }

  Future<Uint8List?> _captureFrameBytes() async {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) return null;
    if (_capturing) return null;

    _capturing = true;
    try {
      final file = await controller.takePicture();
      final bytes = await file.readAsBytes();
      return bytes;
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
      if (!mounted) return;
      setState(() {
        _apiClient?.dispose();
        _apiClient = client;
        _loggedIn = true;
        _status = 'Login successful';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _status = 'Login failed: $e');
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  Future<void> _captureEnrollPhoto() async {
    if (_busy) return;
    final bytes = await _captureFrameBytes();
    if (bytes == null || bytes.isEmpty) {
      setState(() => _status = 'Unable to capture photo.');
      return;
    }
    setState(() {
      _enrollPhotos.add(bytes);
      _status = 'Captured ${_enrollPhotos.length} enrollment photo(s).';
    });
  }

  void _clearEnrollPhotos() {
    setState(() {
      _enrollPhotos.clear();
      _status = 'Enrollment photos cleared.';
    });
  }

  Future<void> _createUserAndRegisterFace() async {
    if (_busy) return;
    if (!_loggedIn || _apiClient == null) {
      setState(() => _status = 'Please login first.');
      return;
    }
    final companyId = int.tryParse(_companyIdCtrl.text.trim());
    if (companyId == null) {
      setState(() => _status = 'Company ID must be a valid number.');
      return;
    }
    if (_enrollPhotos.length < 3) {
      setState(() => _status = 'Capture at least 3 photos before submitting.');
      return;
    }

    setState(() {
      _busy = true;
      _status = 'Creating user and registering face...';
    });

    try {
      final result = await _apiClient!.addUserWithPhotoRequired(
        companyId: companyId,
        name: _newNameCtrl.text.trim(),
        employeeCode: _newEmployeeCodeCtrl.text.trim(),
        password: _newPasswordCtrl.text,
        photos: List<Uint8List>.from(_enrollPhotos),
      );

      if (!mounted) return;
      setState(() {
        _status =
            'User created: id=${result.user.id}, face embeddings saved=${result.faceRegistration.embeddingsSaved}';
        _verifyUserIdCtrl.text = result.user.id.toString();
        _enrollPhotos.clear();
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _status = 'Create/register failed: $e');
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  Future<void> _startScan() async {
    if (_scanning || _busy) return;
    if (!_loggedIn || _apiClient == null) {
      setState(() => _scanStatus = 'Please login first.');
      return;
    }
    final companyId = int.tryParse(_companyIdCtrl.text.trim());
    final userId = int.tryParse(_verifyUserIdCtrl.text.trim());
    if (companyId == null || userId == null) {
      setState(() => _scanStatus = 'Company ID and Verify User ID must be numbers.');
      return;
    }

    _scanner?.dispose();
    _scanner = FaceVerificationScanner(
      apiClient: _apiClient!,
      companyId: companyId,
      userId: userId,
      captureFrameBytes: _captureFrameBytes,
      onResult: (result) {
        if (!mounted) return;
        setState(() {
          _scanStatus =
              'verified=${result.verified} | conf=${result.confidence.toStringAsFixed(3)} | '
              'liveness=${result.livenessScore.toStringAsFixed(3)} | reason=${result.reason}';
          if (result.verified) {
            _verifiedLocked = true;
            _scanning = false;
          }
        });
      },
      onError: (error, _) {
        if (!mounted) return;
        setState(() => _scanStatus = 'Scan error: $error');
      },
    );

    _scanner!.start();
    setState(() {
      _scanning = true;
      _verifiedLocked = false;
      _scanStatus = 'Scanning started...';
    });
  }

  void _stopScan() {
    _scanner?.stop();
    setState(() {
      _scanning = false;
      _scanStatus = 'Scanning stopped.';
    });
  }

  void _retryScan() {
    final scanner = _scanner;
    if (scanner == null) {
      setState(() => _scanStatus = 'Start scanning first.');
      return;
    }
    scanner.retry();
    setState(() {
      _verifiedLocked = false;
      _scanning = true;
      _scanStatus = 'Retry started...';
    });
  }

  @override
  void dispose() {
    _scanner?.dispose();
    _apiClient?.dispose();
    _cameraController?.dispose();

    _baseUrlCtrl.dispose();
    _companyIdCtrl.dispose();
    _adminCodeCtrl.dispose();
    _adminPasswordCtrl.dispose();
    _newNameCtrl.dispose();
    _newEmployeeCodeCtrl.dispose();
    _newPasswordCtrl.dispose();
    _verifyUserIdCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;
    final cameraReady = controller != null && controller.value.isInitialized;

    return Scaffold(
      appBar: AppBar(title: const Text('Face App Integration')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
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
                child: Text(_initializingCamera ? 'Loading camera...' : 'Camera not ready'),
              ),
            const SizedBox(height: 12),
            Text(_status, style: const TextStyle(fontWeight: FontWeight.w600)),
            const SizedBox(height: 12),
            _sectionCard(
              title: '1) Login',
              child: Column(
                children: [
                  TextField(
                    controller: _baseUrlCtrl,
                    decoration: const InputDecoration(labelText: 'Base URL'),
                  ),
                  TextField(
                    controller: _companyIdCtrl,
                    keyboardType: TextInputType.number,
                    decoration: const InputDecoration(labelText: 'Company ID'),
                  ),
                  TextField(
                    controller: _adminCodeCtrl,
                    decoration: const InputDecoration(labelText: 'Admin Employee Code'),
                  ),
                  TextField(
                    controller: _adminPasswordCtrl,
                    obscureText: true,
                    decoration: const InputDecoration(labelText: 'Admin Password'),
                  ),
                  const SizedBox(height: 10),
                  ElevatedButton(
                    onPressed: _busy ? null : _login,
                    child: Text(_loggedIn ? 'Logged In' : 'Login'),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            _sectionCard(
              title: '2) Add User + Face Photos',
              child: Column(
                children: [
                  TextField(
                    controller: _newNameCtrl,
                    decoration: const InputDecoration(labelText: 'New User Name'),
                  ),
                  TextField(
                    controller: _newEmployeeCodeCtrl,
                    decoration: const InputDecoration(labelText: 'New User Employee Code'),
                  ),
                  TextField(
                    controller: _newPasswordCtrl,
                    obscureText: true,
                    decoration: const InputDecoration(labelText: 'New User Password'),
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton(
                          onPressed: _busy || !cameraReady ? null : _captureEnrollPhoto,
                          child: const Text('Capture Photo'),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: OutlinedButton(
                          onPressed: _busy ? null : _clearEnrollPhotos,
                          child: const Text('Clear Photos'),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text('Captured photos: ${_enrollPhotos.length} (minimum 3 required)'),
                  const SizedBox(height: 10),
                  ElevatedButton(
                    onPressed: _busy || !_loggedIn ? null : _createUserAndRegisterFace,
                    child: const Text('Create User + Register Face'),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            _sectionCard(
              title: '3) Verify User by Face Scan',
              child: Column(
                children: [
                  TextField(
                    controller: _verifyUserIdCtrl,
                    keyboardType: TextInputType.number,
                    decoration: const InputDecoration(labelText: 'Verify User ID'),
                  ),
                  const SizedBox(height: 10),
                  Text(_scanStatus),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton(
                          onPressed: _busy || _scanning || _verifiedLocked || !cameraReady ? null : _startScan,
                          child: const Text('Start Scan'),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: OutlinedButton(
                          onPressed: !_scanning ? null : _stopScan,
                          child: const Text('Stop'),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  ElevatedButton(
                    onPressed: !_verifiedLocked ? null : _retryScan,
                    child: const Text('Retry After Verified'),
                  ),
                  const SizedBox(height: 6),
                  const Text(
                    'After successful verification, scan stays stopped until Retry is pressed.',
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _sectionCard({required String title, required Widget child}) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            child,
          ],
        ),
      ),
    );
  }
}
