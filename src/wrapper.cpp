#include <aychip.h>

#include <cstddef>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
using namespace uZX::Chip;


class RegisterWrapper {
public:
    RegisterWrapper(AyumiEmulator& emulator) : AY_(emulator) {}

    auto setR(size_t index, int value) -> void {
        if (index < 0 || index >= std::size(AY_.R)) {
            throw std::out_of_range("Register index out of bounds");
        }
        AY_.R[index] = value;
    }

private:
    AyumiEmulator& AY_;
};


// True if the buffer holds native unsigned 32-bit ints
static auto isUInt32Format(const py::buffer_info& info) -> bool {
    if (info.itemsize != 4) {
        return false;
    }
    std::string fmt = info.format;
    if (!fmt.empty() && (fmt[0] == '@' || fmt[0] == '=')) {
        fmt = fmt.substr(1);
    }
    const uint16_t probe = 1;
    const bool littleEndian = *reinterpret_cast<const uint8_t*>(&probe) == 1;
    if (!fmt.empty() && fmt[0] == (littleEndian ? '<' : '>')) {
        fmt = fmt.substr(1);
    }
    return fmt == "I" || fmt == "L";
}

static inline auto toInt16(double v) -> int16_t {
    v = std::nearbyint(v * 32767.0);
    if (v > 32767.0) return 32767;
    if (v < -32768.0) return -32768;
    return static_cast<int16_t>(v);
}

static inline auto putInt16LE(char* dst, int16_t v) -> void {
    const uint16_t u = static_cast<uint16_t>(v);
    dst[0] = static_cast<char>(u & 0xff);
    dst[1] = static_cast<char>(u >> 8);
}

struct RegWrite {
    uint64_t sample;  // write is applied before this sample is rendered
    uint8_t reg;
    uint8_t value;
};

static auto renderFrame(AyumiEmulator& AY, const py::buffer& writes, int samples, int64_t tFrame,
                        bool stereo, bool removeDC) -> py::bytes {
    if (samples <= 0) {
        throw std::invalid_argument("samples must be greater than 0");
    }
    if (tFrame <= 0) {
        throw std::invalid_argument("t_frame must be greater than 0");
    }
    auto info = writes.request();
    if (!isUInt32Format(info)) {
        throw std::invalid_argument("writes buffer format must be uint32 (array('I') or numpy.uint32), got '"
                                    + info.format + "'");
    }
    size_t count;
    if (info.ndim == 1) {
        if (info.shape[0] % 3 != 0) {
            throw std::invalid_argument("flat writes buffer length must be a multiple of 3");
        }
        if (info.shape[0] > 1 && info.strides[0] != 4) {
            throw std::invalid_argument("writes buffer must be contiguous");
        }
        count = static_cast<size_t>(info.shape[0] / 3);
    } else if (info.ndim == 2) {
        if (info.shape[1] != 3) {
            throw std::invalid_argument("2-D writes buffer must have shape (n, 3)");
        }
        if (info.shape[0] > 0 && (info.strides[1] != 4 || (info.shape[0] > 1 && info.strides[0] != 12))) {
            throw std::invalid_argument("writes buffer must be contiguous");
        }
        count = static_cast<size_t>(info.shape[0]);
    } else {
        throw std::invalid_argument("writes buffer must be 1-D (3n) or 2-D (n, 3)");
    }

    // Validate everything and copy before touching the chip state
    const uint32_t* src = static_cast<const uint32_t*>(info.ptr);
    std::vector<RegWrite> list(count);
    uint64_t prevT = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint64_t t = src[i * 3];
        const uint32_t reg = src[i * 3 + 1];
        const uint32_t value = src[i * 3 + 2];
        if (t < prevT) {
            throw std::invalid_argument("write " + std::to_string(i) + ": t must be non-decreasing");
        }
        if (t >= static_cast<uint64_t>(tFrame)) {
            throw std::invalid_argument("write " + std::to_string(i) + ": t must be less than t_frame");
        }
        if (reg >= std::size(AY.R)) {
            throw std::invalid_argument("write " + std::to_string(i) + ": register must be in 0..13");
        }
        if (value > 255) {
            throw std::invalid_argument("write " + std::to_string(i) + ": value must be in 0..255");
        }
        prevT = t;
        // k = ceil(t * samples / t_frame), exact in 64-bit
        const uint64_t k = (t * static_cast<uint64_t>(samples) + static_cast<uint64_t>(tFrame) - 1)
                           / static_cast<uint64_t>(tFrame);
        list[i] = RegWrite {k, static_cast<uint8_t>(reg), static_cast<uint8_t>(value)};
    }

    const size_t n = static_cast<size_t>(samples);
    const size_t outCount = stereo ? 2 * n : n;
    py::bytes result(nullptr, outCount * sizeof(int16_t));
    char* out = PyBytes_AS_STRING(result.ptr());

    {
        py::gil_scoped_release release;
        std::vector<float> left(n), right(n);
        size_t pos = 0;
        for (const auto& w : list) {
            if (w.sample > pos) {
                AY.processBlock(left.data() + pos, right.data() + pos, w.sample - pos, removeDC);
                pos = w.sample;
            }
            AY.R[w.reg] = w.value;
        }
        if (pos < n) {
            AY.processBlock(left.data() + pos, right.data() + pos, n - pos, removeDC);
        }
        for (size_t i = 0; i < n; ++i) {
            const double l = left[i];
            const double r = right[i];
            if (stereo) {
                putInt16LE(out + 4 * i, toInt16(l));
                putInt16LE(out + 4 * i + 2, toInt16(r));
            } else {
                putInt16LE(out + 2 * i, toInt16((l + r) * 0.5));
            }
        }
    }
    return result;
}


PYBIND11_MODULE(pyayay, m) {
    m.doc() = "Python bindings for Ayumi sound chip emulator";

    py::enum_<AYInterface::TypeEnum::Enum>(m, "ChipType")
        .value("AY", AYInterface::TypeEnum::AY, "AY-3-8910")
        .value("YM", AYInterface::TypeEnum::YM, "YM2149")
        .export_values();

    py::enum_<AYInterface::EnvShapeEnum::Enum>(m, "EnvShape")
        .value("DOWN_HOLD_BOTTOM_0", AYInterface::EnvShapeEnum::DOWN_HOLD_BOTTOM_0, "\\___" )
        .value("DOWN_HOLD_BOTTOM_1", AYInterface::EnvShapeEnum::DOWN_HOLD_BOTTOM_1, "\\___" )
        .value("DOWN_HOLD_BOTTOM_2", AYInterface::EnvShapeEnum::DOWN_HOLD_BOTTOM_2, "\\___" )
        .value("DOWN_HOLD_BOTTOM_3", AYInterface::EnvShapeEnum::DOWN_HOLD_BOTTOM_3, "\\___" )
        .value("UP_HOLD_BOTTOM_4",   AYInterface::EnvShapeEnum::UP_HOLD_BOTTOM_4,   "/|__"  )
        .value("UP_HOLD_BOTTOM_5",   AYInterface::EnvShapeEnum::UP_HOLD_BOTTOM_5,   "/|__"  )
        .value("UP_HOLD_BOTTOM_6",   AYInterface::EnvShapeEnum::UP_HOLD_BOTTOM_6,   "/|__"  )
        .value("UP_HOLD_BOTTOM_7",   AYInterface::EnvShapeEnum::UP_HOLD_BOTTOM_7,   "/|__"  )
        .value("DOWN_DOWN_8",        AYInterface::EnvShapeEnum::DOWN_DOWN_8,        "\\|\\|")
        .value("DOWN_HOLD_BOTTOM_9", AYInterface::EnvShapeEnum::DOWN_HOLD_BOTTOM_9, "\\___" )
        .value("DOWN_UP_A",          AYInterface::EnvShapeEnum::DOWN_UP_A,          "\\/\\/")
        .value("DOWN_HOLD_TOP_B",    AYInterface::EnvShapeEnum::DOWN_HOLD_TOP_B,    "\\|~~" )
        .value("UP_UP_C",            AYInterface::EnvShapeEnum::UP_UP_C,            "/|/|"  )
        .value("UP_HOLD_TOP_D",      AYInterface::EnvShapeEnum::UP_HOLD_TOP_D,      "/~~~~" )
        .value("UP_DOWN_E",          AYInterface::EnvShapeEnum::UP_DOWN_E,          "/\\/\\")
        .value("UP_HOLD_BOTTOM_F",   AYInterface::EnvShapeEnum::UP_HOLD_BOTTOM_F,   "/|__"  )
        .export_values();

    py::class_<RegisterWrapper>(m, "Register")
        .def(py::init<AyumiEmulator&>())
        .def("__setitem__", &RegisterWrapper::setR)
        ;

    py::class_<AyumiEmulator>(m, "Ayumi")
        .def_property_readonly_static("AY", [](py::object) { return AYInterface::TypeEnum::AY; })
        .def_property_readonly_static("YM", [](py::object) { return AYInterface::TypeEnum::YM; })

        .def(py::init<int, double, AYInterface::TypeEnum::Enum>(),
             py::arg("sample_rate") = 44100,
             py::arg("clock") = 1773400,
             py::arg("type") = AYInterface::TypeEnum::AY
        )
        .def_property_readonly("R", [](AyumiEmulator& AY) { return RegisterWrapper(AY); },
              py::return_value_policy::reference_internal)

        .def("set_registers", [](AyumiEmulator& AY, const std::vector<uint8_t>& regs, const std::vector<uint8_t>& values) {
            if (regs.size() != values.size()) {
                throw std::invalid_argument("Buffer sizes must match");
            }
            for (size_t i = 0; i < regs.size(); ++i) {
                if (regs[i] < 0 || regs[i] >= std::size(AY.R)) {
                    throw std::out_of_range("Register index out of bounds");
                }
                AY.R[regs[i]] = values[i];
            }
        }, py::arg("registers"), py::arg("values"))

        .def("set_registers_masked", [](AyumiEmulator& AY, const py::buffer& values, const py::buffer& mask) {
            auto maskInfo = mask.request();
            auto valuesInfo = values.request();
            if (maskInfo.ndim != 1 || valuesInfo.ndim != 1) {
                throw std::invalid_argument("Incompatible buffers dimension, must be 1");
            }
            if (valuesInfo.size != static_cast<int>(std::size(AY.R))) {
                throw std::invalid_argument("Values size must match number of registers (14)");
            }
            if (maskInfo.size != valuesInfo.size) {
                throw std::invalid_argument("Buffer sizes must match");
            }
            if (valuesInfo.format != py::format_descriptor<uint8_t>::format()) {
                throw std::invalid_argument("Values buffer format must be uint8_t");
            }
            if (maskInfo.format != py::format_descriptor<bool>::format()) {
                throw std::invalid_argument("Mask buffer format must be bool");
            }
            if (maskInfo.strides[0] != sizeof(bool) || valuesInfo.strides[0] != sizeof(uint8_t)) {
                throw std::invalid_argument("Buffers must be contiguous");
            }
            uint8_t* maskPtr = static_cast<uint8_t*>(maskInfo.ptr);
            bool* valuesPtr = static_cast<bool*>(valuesInfo.ptr);
            for (int i = 0; i < maskInfo.size; ++i) {
                if (!maskPtr[i]) {
                    AY.R[i] = valuesPtr[i];
                }
            }
        }, py::arg("values"), py::arg("mask"), "Set registers with mask. Mask is a list of 14 bytes, 0 means do not change register, >0 means change register")

        .def("render_psg", [](AyumiEmulator& AY, const py::buffer& psg, const py::buffer& mask,
                              py::buffer outLeft, py::buffer outRight, float fps, bool remove_dc) {
            auto psgInfo = psg.request();
            auto maskInfo = mask.request();
            auto outLeftInfo = outLeft.request();
            auto outRightInfo = outRight.request();
            if (outLeftInfo.ndim != 1 || outRightInfo.ndim != 1) {
                throw std::invalid_argument("Incompatible buffers dimension, must be 1");
            }
            if (outLeftInfo.size != outRightInfo.size) {
                throw std::invalid_argument("Buffer sizes must match");
            }
            if (outLeftInfo.format != py::format_descriptor<float>::format() || outRightInfo.format != py::format_descriptor<float>::format()) {
                throw std::runtime_error("Buffer format must be float");
            }
            if (outLeftInfo.strides[0] != sizeof(float) || outRightInfo.strides[0] != sizeof(float)) {
                throw std::runtime_error("Output buffers must be contiguous");
            }
            if (maskInfo.ndim != 2 || psgInfo.ndim != 2) {
                throw std::invalid_argument("Incompatible buffers dimension, must be 2");
            }
            if (psgInfo.shape[1] != static_cast<int>(std::size(AY.R))) {
                throw std::invalid_argument("Values dim 1 must match number of registers (14)");
            }
            if (maskInfo.shape[1] != static_cast<int>(std::size(AY.R))) {
                throw std::invalid_argument("Mask dim 1 must match number of registers (14)");
            }
            if (maskInfo.shape[0] != psgInfo.shape[0]) {
                throw std::invalid_argument("Buffer sizes must match");
            }
            if (psgInfo.format != py::format_descriptor<uint8_t>::format()) {
                throw std::invalid_argument("Values buffer format must be uint8_t");
            }
            if (maskInfo.format != py::format_descriptor<bool>::format()) {
                throw std::invalid_argument("Mask buffer format must be bool");
            }
            if (maskInfo.strides[1] != sizeof(bool) || psgInfo.strides[1] != sizeof(uint8_t)) {
                throw std::invalid_argument("PSG buffers must be contiguous");
            }
            float duration_sec = psgInfo.shape[0] / fps;
            int samples = static_cast<int>(std::ceil(duration_sec * AY.getSampleRate()));
            float samples_per_frame = static_cast<float>(AY.getSampleRate()) / fps;
            if (outLeftInfo.size < samples || outRightInfo.size < samples) {
                throw std::invalid_argument("Buffer sizes must be at least" + std::to_string(samples)
                                         + " got " + std::to_string(outLeftInfo.size));
            }
            float* outLeftPtr = static_cast<float*>(outLeftInfo.ptr);
            float* outRightPtr = static_cast<float*>(outRightInfo.ptr);
            const int stride = 1;
            uint8_t* psgPtr = static_cast<uint8_t*>(psgInfo.ptr);
            uint8_t* maskPtr = static_cast<uint8_t*>(maskInfo.ptr);
            for (size_t i = 0; i < static_cast<size_t>(psgInfo.shape[0]); ++i) {
                for (size_t j = 0; j < std::size(AY.R); ++j) {
                    if (!maskPtr[i * std::size(AY.R) + j]) {
                        AY.R[j] = psgPtr[i * std::size(AY.R) + j];
                    }
                }
                const size_t sample_begin_frame = std::round(i * samples_per_frame);
                const size_t sample_end_frame = std::round((i + 1) * samples_per_frame);
                const size_t samples_to_render = sample_end_frame - sample_begin_frame;
                AY.processBlock(outLeftPtr, outRightPtr, samples_to_render, remove_dc, stride);
                outLeftPtr += samples_to_render;
                outRightPtr += samples_to_render;
            }
        }, py::arg("psg"), py::arg("mask"), py::arg("out_left"), py::arg("out_right"), py::arg("fps"), py::arg("remove_dc") = true)

        .def("process_block", [](AyumiEmulator& AY, py::buffer outLeft, py::buffer outRight, int samples, bool remove_dc) {
            auto outLeftInfo = outLeft.request();
            auto outRightInfo = outRight.request();
            if (outLeftInfo.ndim != 1 || outRightInfo.ndim != 1) {
                throw std::invalid_argument("Incompatible buffers dimension, must be 1");
            }
            if (outLeftInfo.size != outRightInfo.size) {
                throw std::invalid_argument("Buffer sizes must match");
            }
            if (outLeftInfo.format != py::format_descriptor<float>::format() || outRightInfo.format != py::format_descriptor<float>::format()) {
                throw std::invalid_argument("Buffer format must be float");
            }
            if (outLeftInfo.strides[0] != sizeof(float) || outRightInfo.strides[0] != sizeof(float)) {
                throw std::invalid_argument("Buffers must be contiguous");
            }
            if (outLeftInfo.size < samples || outRightInfo.size < samples) {
                throw std::invalid_argument("Buffer sizes must be at least" + std::to_string(samples)
                                         + " got " + std::to_string(outLeftInfo.size));
            }
            if (samples <= 0) {
                throw std::invalid_argument("Samples must be greater than 0");
            }
            float* outLeftPtr = static_cast<float*>(outLeftInfo.ptr);
            float* outRightPtr = static_cast<float*>(outRightInfo.ptr);
            const int stride = 1;
            AY.processBlock(outLeftPtr, outRightPtr, samples, remove_dc, stride);
        }, py::arg("out_left"), py::arg("out_right"), py::arg("samples"), py::arg("remove_dc") = true)

        .def("render_frame", &renderFrame,
             py::arg("writes"), py::arg("samples"), py::arg("t_frame"),
             py::arg("stereo") = false, py::arg("remove_dc") = true,
             "Render one frame to little-endian int16 PCM bytes, applying timed register writes.\n"
             "writes: uint32 buffer of (t, reg, value) triples, flat (3n) or (n, 3), t non-decreasing.\n"
             "A write at t is applied before sample ceil(t * samples / t_frame).")

        .def("reset", [](AyumiEmulator& AY, int sampleRate, double clock, AYInterface::TypeEnum::Enum type) {
            AY.Reset(sampleRate, clock, type);
            },
            py::arg("sample_rate") = 44100,
            py::arg("clock") = 1773400.0,
            py::arg("type") = AYInterface::TypeEnum::AY
        )
        .def("can_change_clock", &AyumiEmulator::canChangeClock)
        .def("can_change_clock_continously", &AyumiEmulator::canChangeClockContinously)
        .def("get_clock_values", &AyumiEmulator::getClockValues)
        .def("set_sample_rate", &AyumiEmulator::setSampleRate, py::arg("sampleRate"))
        .def("get_sample_rate", &AyumiEmulator::getSampleRate)

        .def("set_type", [](AyumiEmulator& AY, AYInterface::TypeEnum::Enum type) {
            AY.setType(type);
        }, py::arg("type"))
        .def("get_type", [](AyumiEmulator& AY) {
            return static_cast<AYInterface::TypeEnum::Enum>(AY.getType()); })

        .def("get_clock", &AyumiEmulator::getClock)
        .def("set_clock", &AyumiEmulator::setClock, py::arg("rate"))

        .def("set_pan", &AyumiEmulator::setPan,
            py::arg("index"), py::arg("value"), py::arg("is_eqp") = false)
        .def("get_pan", &AyumiEmulator::getPan, py::arg("index"))

        .def("set_tone_period", &AyumiEmulator::setTonePeriod, py::arg("index"), py::arg("period"))
        .def("get_tone_period", &AyumiEmulator::getTonePeriod, py::arg("index"))
        .def("set_mixer",  &AyumiEmulator::setMixer, py::arg("index"), py::arg("tone"), py::arg("noise"), py::arg("envelope"))
        .def("set_volume", &AyumiEmulator::setVolume, py::arg("index"), py::arg("volume"))
        .def("get_volume", &AyumiEmulator::getVolume, py::arg("index"))
        .def("set_envelope_period", &AyumiEmulator::setEnvelopePeriod, py::arg("period"))
        .def("get_envelope_period", &AyumiEmulator::getEnvelopePeriod)

        .def("set_envelope_shape", [](AyumiEmulator& AY, AYInterface::EnvShapeEnum::Enum shape) {
                AY.setEnvelopeShape(shape);
            }, py::arg("shape"))
        .def("set_envelope_shape", [](AyumiEmulator& AY, int shape) {
                AY.setEnvelopeShape(shape);
            }, py::arg("shape"))
        .def("get_envelope_shape", [](const AyumiEmulator& AY) {
                return static_cast<AYInterface::EnvShapeEnum::Enum>(AY.getEnvelopeShape());
            })

        .def("set_noise_period", &AyumiEmulator::setNoisePeriod, py::arg("period"))
        .def("get_noise_period", &AyumiEmulator::getNoisePeriod)

        .def("set_master_volume", &AyumiEmulator::setMasterVolume, py::arg("volume"))
        .def("get_master_volume", &AyumiEmulator::getMasterVolume)

        .def("__copy__", [](const AyumiEmulator& AY) {
            return AyumiEmulator(AY);
        }, py::return_value_policy::copy)
        .def("copy", [](const AyumiEmulator& AY) {
            return AyumiEmulator(AY);
        }, py::return_value_policy::copy)
        ;
}
