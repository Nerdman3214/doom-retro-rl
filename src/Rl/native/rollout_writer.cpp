#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" {

struct RolloutRecord {
    uint8_t frame[3 * 84 * 84];
    uint8_t action;
    uint8_t done;
    int16_t health;
    int16_t ammo;
    float reward;
    float x;
    float y;
    uint32_t flags;
};

FILE* open_rollout_file(const char* path) {
    return std::fopen(path, "ab");
}

int write_rollout_record(
    FILE* file,
    const uint8_t* frame,
    uint8_t action,
    uint8_t done,
    int16_t health,
    int16_t ammo,
    float reward,
    float x,
    float y,
    uint32_t flags
) {
    if (!file || !frame) {
        return 0;
    }

    RolloutRecord record;
    std::memcpy(record.frame, frame, 3 * 84 * 84);

    record.action = action;
    record.done = done;
    record.health = health;
    record.ammo = ammo;
    record.reward = reward;
    record.x = x;
    record.y = y;
    record.flags = flags;

    size_t written = std::fwrite(&record, sizeof(RolloutRecord), 1, file);
    return written == 1;
}

void close_rollout_file(FILE* file) {
    if (file) {
        std::fclose(file);
    }
}

}