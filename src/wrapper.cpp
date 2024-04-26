#include <aychip.h>

#include <cstddef>
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


PYBIND11_MODULE(pyayay, m) {
    m.doc() = "Python bindings for Ayumi sound chip emulator";

    py::enum_<AYInterface::Enum>(m, "ChipType")
        .value("AY", AYInterface::AY, "AY-3-8910")
        .value("YM", AYInterface::YM, "YM2149")
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
        .def_property_readonly_static("AY", [](py::object) { return AYInterface::AY; })
        .def_property_readonly_static("YM", [](py::object) { return AYInterface::YM; })

        .def(py::init<int, double, AYInterface::Enum>(),
             py::arg("sample_rate") = 44100,
             py::arg("clock") = 1773400,
             py::arg("type") = AYInterface::AY
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
                AY.processBlock(outLeftPtr, outRightPtr, samples_to_render, stride, remove_dc);
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
            AY.processBlock(outLeftPtr, outRightPtr, samples, stride, remove_dc);
        }, py::arg("out_left"), py::arg("out_right"), py::arg("samples"), py::arg("remove_dc") = true)

        .def("reset", [](AyumiEmulator& AY, int sampleRate, double clock, AYInterface::Enum type) {
            AY.Reset(sampleRate, clock, type);
            },
            py::arg("sample_rate") = 44100,
            py::arg("clock") = 1773400.0,
            py::arg("type") = AYInterface::AY
        )
        .def("can_change_clock", &AyumiEmulator::canChangeClock)
        .def("can_change_clock_continously", &AyumiEmulator::canChangeClockContinously)
        .def("get_clock_values", &AyumiEmulator::getClockValues)
        .def("set_sample_rate", &AyumiEmulator::setSampleRate, py::arg("sampleRate"))
        .def("get_sample_rate", &AyumiEmulator::getSampleRate)

        .def("set_type", [](AyumiEmulator& AY, AYInterface::Enum type) {
            AY.setType(type);
        }, py::arg("type"))
        .def("get_type", [](AyumiEmulator& AY) {
            return static_cast<AYInterface::Enum>(AY.getType()); })

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
