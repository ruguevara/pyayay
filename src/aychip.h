#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "utils/tools.h"

namespace uZX::Chip {

/*****************************************************************************/
/*  C++ wrapper for aychip struct and functions                              */
/*****************************************************************************/

namespace {
    extern "C" {
        #include "ayumi/ayumi.h"
    }
}


// We must divide AYChip into an interface and implementation, because implementation can vary or even be drivers for real hardware chips
// Ayumi AY/YM
// Other software emulated AY/YM
// Hardware AY/YM
// Realtime digital (maybe midi) link to a real hardware computer with some client
// Realtime software link to emulated computer with some client

class AYInterface {
public:
    struct TypeEnum {
        enum Enum {
            AY,
            YM
        };
        static inline constexpr std::string_view labels[] {
            "AY",
            "YM"
        };
    };
    using ChipType = EnumChoice<TypeEnum>;

    struct EnvShapeEnum {
        enum Enum {
            DOWN_HOLD_BOTTOM_0,
            DOWN_HOLD_BOTTOM_1,
            DOWN_HOLD_BOTTOM_2,
            DOWN_HOLD_BOTTOM_3,
            UP_HOLD_BOTTOM_4,
            UP_HOLD_BOTTOM_5,
            UP_HOLD_BOTTOM_6,
            UP_HOLD_BOTTOM_7,
            DOWN_DOWN_8,
            DOWN_HOLD_BOTTOM_9,
            DOWN_UP_A,
            DOWN_HOLD_TOP_B,
            UP_UP_C,
            UP_HOLD_TOP_D,
            UP_DOWN_E,
            UP_HOLD_BOTTOM_F,
        };
        static inline constexpr std::string_view labels[] {
            "\\___",
            "\\___",
            "\\___",
            "\\___",
            "/|__",
            "/|__",
            "/|__",
            "/|__",
            "\\|\\|",
            "\\___",
            "\\/\\/",
            "\\|~~",
            "/|/|",
            "/~~~~",
            "/\\/\\",
            "/|__",
        };
    };
    using EnvShape = EnumChoice<EnvShapeEnum>;

    // AYInterface(int sampleRate = 44100, double clock = 2000000, ChipType type = TypeEnum::AY) ;
    virtual ~AYInterface() {};

    virtual auto canChangeClock() const -> bool = 0;
    virtual auto canChangeClockContinously() const -> bool = 0;
    virtual auto getClockValues() const -> std::vector<float> = 0;
    virtual auto setSampleRate(int sampleRate) -> void = 0;
    virtual auto getSampleRate() const -> int = 0;
    virtual auto setType(ChipType type) -> void = 0;
    virtual auto getType() const -> ChipType = 0;
    virtual auto getClock() const -> double = 0;
    virtual auto setClock(double v) -> void = 0;
    virtual auto setPan(int chan, double pan, bool isEqp = false) -> void = 0;
    virtual auto getPan(int chan) const -> double = 0;
    virtual auto setMixer(int chan, bool tOn, bool nOn, bool eOn) -> void = 0;
    virtual auto setEnvelopeOn(int chan, bool on) -> void = 0;
    virtual auto setNoiseOn(int chan, bool on) -> void = 0;
    virtual auto setVolume(int chan, int volume) -> void = 0;
    virtual auto getVolume(int chan) const -> int = 0;
    virtual auto setToneOn(int chan, bool on) -> void = 0;
    virtual auto setTonePeriod(int chan, int period) -> void = 0;
    virtual auto getTonePeriod(int chan) const -> int = 0;
    virtual auto setNoisePeriod(int period) -> void = 0;
    virtual auto getNoisePeriod() const -> int = 0;
    virtual auto setEnvelopeShape(EnvShape shape) -> void = 0;
    virtual auto getEnvelopeShape() const -> EnvShape = 0;
    virtual auto setEnvelopePeriod(int period) -> void = 0;
    virtual auto getEnvelopePeriod() const -> int = 0;
    virtual auto setMasterVolume(float volume) -> void = 0;
    virtual auto getMasterVolume() const -> float = 0;
    virtual auto processBlock(float* outLeft, float* outRight, size_t numSamples, bool removeDC = true, size_t stride = 1) -> void = 0;

private:
    inline void setFineTonePeriod(int chan, unsigned char fine) noexcept {
        const int oldPeriod = getTonePeriod(chan);
        setTonePeriod(chan, (oldPeriod & 0xff00) | (fine & 0xff));
    }
    inline void setCoarseTonePeriod(int chan, unsigned char coarse) noexcept {
        const int oldPeriod = getTonePeriod(chan);
        setTonePeriod(chan, (oldPeriod & 0xff) | (coarse << 8));
    }
    // TODO Actually to be able to record and set R0-R13 registers
    // we need to do things in reverse: setPeriod, setMixer, setVolume,
    // setEnvelope, setNoise, setTone should work with R0-R13 registers
    // And AY implementation should implement functions setR0-R13 or
    // setFineTonePeriod, setCoarseTonePeriod, setNoisePeriod, setMixer, setVolume, setEnvelope
    inline void setR0(unsigned char finePeriodA)   noexcept { setFineTonePeriod  (0, finePeriodA); }
    inline void setR1(unsigned char coarsePeriodA) noexcept { setCoarseTonePeriod(0, coarsePeriodA); }
    inline void setR2(unsigned char finePeriodB)   noexcept { setFineTonePeriod  (1, finePeriodB); }
    inline void setR3(unsigned char coarsePeriodB) noexcept { setCoarseTonePeriod(1, coarsePeriodB); }
    inline void setR4(unsigned char finePeriodC)   noexcept { setFineTonePeriod  (2, finePeriodC); }
    inline void setR5(unsigned char coarsePeriodC) noexcept { setCoarseTonePeriod(2, coarsePeriodC); }
    inline void setR6(unsigned char noisePeriod)   noexcept { setNoisePeriod     (   noisePeriod); }
    inline void setR7(unsigned char mixer) noexcept {
        //   7   |   6   |    5    |    4    |    3    |   2    |   1    |   0
        // I/O B | I/O A | Noise C | Noise B | Noise A | Tone C | Tone B | Tone A
        const bool Atone = !(mixer & 1);
        const bool Btone = !((mixer >> 1) & 1);
        const bool Ctone = !((mixer >> 2) & 1);
        const bool Anoise = !((mixer >> 3) & 1);
        const bool Bnoise = !((mixer >> 4) & 1);
        const bool Cnoise = !((mixer >> 5) & 1);
        setToneOn(0, Atone);
        setToneOn(1, Btone);
        setToneOn(2, Ctone);
        setNoiseOn(0, Anoise);
        setNoiseOn(1, Bnoise);
        setNoiseOn(2, Cnoise);
    }
    inline void setR8 (unsigned char volumeA) noexcept { setVolume(0, volumeA & 0x0f); setEnvelopeOn(0, volumeA & 0x10); }
    inline void setR9 (unsigned char volumeB) noexcept { setVolume(1, volumeB & 0x0f); setEnvelopeOn(1, volumeB & 0x10); }
    inline void setR10(unsigned char volumeC) noexcept { setVolume(2, volumeC & 0x0f); setEnvelopeOn(2, volumeC & 0x10); }
    inline void setR11(unsigned char envFinePeriod) noexcept {
        const int oldPeriod = getEnvelopePeriod();
        setEnvelopePeriod((oldPeriod & 0xff00) | (envFinePeriod & 0xff));
    }
    inline void setR12(unsigned char envCoarsePeriod) noexcept {
        const int oldPeriod = getEnvelopePeriod();
        setEnvelopePeriod((oldPeriod & 0xff) | (envCoarsePeriod << 8));
    }
    inline void setR13(unsigned char envShape) noexcept {
        setEnvelopeShape(static_cast<EnvShape>(envShape));
    }

    /************************************************************************/
    /* Register array accessor structure. Example of usage:                 */
    /*   uint8_t v = ay.R[0]                                                */
    /*   ay.R[0] = 42                                                       */
    /************************************************************************/

public:
    typedef void (AYInterface::*SetterFunction)(unsigned char);

    class RegisterAccessor {
    private:
        AYInterface& Obj_;
        SetterFunction Setter_;

    public:
        void operator=(int value) {
            (Obj_.*Setter_)(value);
        }
        RegisterAccessor(AYInterface& obj, SetterFunction setter)
            : Obj_(obj)
            , Setter_(setter)
        {}
    };

    AYInterface()
        : R {
            RegisterAccessor {*this, &AYInterface::setR0},
            RegisterAccessor {*this, &AYInterface::setR1},
            RegisterAccessor {*this, &AYInterface::setR2},
            RegisterAccessor {*this, &AYInterface::setR3},
            RegisterAccessor {*this, &AYInterface::setR4},
            RegisterAccessor {*this, &AYInterface::setR5},
            RegisterAccessor {*this, &AYInterface::setR6},
            RegisterAccessor {*this, &AYInterface::setR7},
            RegisterAccessor {*this, &AYInterface::setR8},
            RegisterAccessor {*this, &AYInterface::setR9},
            RegisterAccessor {*this, &AYInterface::setR10},
            RegisterAccessor {*this, &AYInterface::setR11},
            RegisterAccessor {*this, &AYInterface::setR12},
            RegisterAccessor {*this, &AYInterface::setR13}
        }
    {};

    std::array<RegisterAccessor, 14> R;
};


class AyumiEmulator : public AYInterface {
public:
    AyumiEmulator(int sampleRate = 44100, double clock = 2000000, ChipType type = TypeEnum::YM);
    ~AyumiEmulator() override;
    auto Reset(int sampleRate = 44100, double clock = 2000000, ChipType type = TypeEnum::YM) -> void;

    auto canChangeClock() const -> bool override;
    auto canChangeClockContinously() const -> bool override;
    auto getClock() const -> double override;
    auto getClockValues() const -> std::vector<float> override;
    auto setSampleRate(int sampleRate) -> void override;
    auto getSampleRate() const -> int override;
    auto setType(ChipType type) -> void override;
    auto getType() const -> ChipType override;
    auto setClock(double v) -> void override;
    auto setPan(int chan, double pan, bool isEqp = false) -> void override;
    auto getPan(int chan) const -> double override;
    auto setMixer(int chan, bool tOn, bool nOn, bool eOn) -> void override;
    auto setEnvelopeOn(int chan, bool on) -> void override;
    auto setNoiseOn(int chan, bool on) -> void override;
    auto setVolume(int chan, int volume) -> void override;
    auto getVolume(int chan) const -> int override;
    auto setToneOn(int chan, bool on) -> void override;
    auto setTonePeriod(int chan, int period) -> void override;
    auto getTonePeriod(int chan) const -> int override;
    auto setNoisePeriod(int period) -> void override;
    auto getNoisePeriod() const -> int override;
    auto setEnvelopeShape(EnvShape shape) -> void override;
    auto getEnvelopeShape() const -> EnvShape override;
    auto setEnvelopePeriod(int period) -> void override;
    auto getEnvelopePeriod() const -> int override;
    auto setMasterVolume(float volume) -> void override;
    auto getMasterVolume() const -> float override;
    auto processBlock(float* outLeft, float* outRight, size_t numSamples, bool removeDC = true, size_t stride = 1) -> void override;
    // TODO
    // * Output to thee separate channels instead of mixing them to stereo panorama

private:
    ayumi Ayumi_;
    ChipType Type_;
    double ClockRate_;
    int SampleRate_;
    double Pan_[TONE_CHANNELS];
    float MasterVolume_;
};

} // namespace uZX::Chip
