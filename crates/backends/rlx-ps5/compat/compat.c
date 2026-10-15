// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// libc functions the prebuilt FreeBSD std imports but the PS5's libraries do
// not export. Linked into every payload by build.rs.

#include <dirent.h>
#include <fcntl.h>
#include <link.h>
#include <pthread.h>
#include <pthread_np.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

// std names its threads with the POSIX-2008 call; Sony's libkernel only has
// FreeBSD's original `pthread_set_name_np`.
int pthread_setname_np(pthread_t thread, const char *name) {
    pthread_set_name_np(thread, name);
    return 0;
}

// std's `TcpListener::accept` calls `accept4(..., SOCK_CLOEXEC)`. The SDK's
// headers declare it (FreeBSD 10+), but libkernel only exports `accept`.
int accept4(int fd, struct sockaddr *addr, socklen_t *len, int flags) {
    int c = accept(fd, addr, len);
    if (c < 0) {
        return c;
    }
    if (((flags & SOCK_CLOEXEC) && fcntl(c, F_SETFD, FD_CLOEXEC) < 0) ||
        ((flags & SOCK_NONBLOCK) && fcntl(c, F_SETFL, fcntl(c, F_GETFL) | O_NONBLOCK) < 0)) {
        close(c);
        return -1;
    }
    return c;
}

// The SDK's linker script (ldscripts/elf_x86_64.x) marks the image bounds and
// the unwind table; the payload is one PIE module linked at 0, so the load
// bias is where `__image_start` landed.
extern const char __image_start[] __attribute__((visibility("hidden")));
extern const char __image_end[] __attribute__((visibility("hidden")));
extern const char __eh_frame_hdr_start[] __attribute__((visibility("hidden")));
extern const char __eh_frame_hdr_end[] __attribute__((visibility("hidden")));

// std's backtrace symbolizer walks the loaded modules. There is no rtld list
// for a payload, so report the payload itself: one load segment spanning the
// image plus its PT_GNU_EH_FRAME.
int dl_iterate_phdr(int (*cb)(struct dl_phdr_info *, size_t, void *), void *data) {
    static Elf_Phdr phdr[2];
    const Elf_Addr base = (Elf_Addr)__image_start;
    phdr[0].p_type = PT_LOAD;
    phdr[0].p_flags = PF_R | PF_X;
    phdr[0].p_vaddr = 0;
    phdr[0].p_memsz = (Elf_Addr)__image_end - base;
    phdr[0].p_filesz = phdr[0].p_memsz;
    phdr[1].p_type = PT_GNU_EH_FRAME;
    phdr[1].p_flags = PF_R;
    phdr[1].p_vaddr = (Elf_Addr)__eh_frame_hdr_start - base;
    phdr[1].p_memsz = (Elf_Addr)__eh_frame_hdr_end - (Elf_Addr)__eh_frame_hdr_start;
    phdr[1].p_filesz = phdr[1].p_memsz;

    struct dl_phdr_info info = {0};
    info.dlpi_addr = base;
    info.dlpi_name = "";
    info.dlpi_phdr = phdr;
    info.dlpi_phnum = 2;
    info.dlpi_adds = 1;
    return cb(&info, sizeof(info), data);
}

// ── FreeBSD 11 → 12 `stat` / `dirent` ──────────────────────────────────────
//
// The console's libc is FreeBSD 11 (32-bit ino_t, 16-bit nlink_t). Rust's std
// only builds against the libc crate's FreeBSD 12 layouts (64-bit inodes), so
// its calls are redirected here with `ld --wrap` (scripts/ps5-cc): each wrapper
// calls Sony's function with the FreeBSD 11 struct and widens the result into
// the caller's FreeBSD 12 one. Layouts mirror libc's `freebsd12` module.

struct stat12 {
    uint64_t st_dev;
    uint64_t st_ino;
    uint64_t st_nlink;
    uint16_t st_mode;
    int16_t st_padding0;
    uint32_t st_uid;
    uint32_t st_gid;
    int32_t st_padding1;
    uint64_t st_rdev;
    struct timespec st_atim;
    struct timespec st_mtim;
    struct timespec st_ctim;
    struct timespec st_birthtim;
    int64_t st_size;
    int64_t st_blocks;
    int32_t st_blksize;
    uint32_t st_flags;
    uint64_t st_gen;
    uint64_t st_spare[10];
};
_Static_assert(sizeof(struct stat12) == 224, "FreeBSD 12 struct stat is 224 bytes");

struct dirent12 {
    uint64_t d_fileno;
    int64_t d_off;
    uint16_t d_reclen;
    uint8_t d_type;
    uint8_t d_pad0;
    uint16_t d_namlen;
    uint16_t d_pad1;
    char d_name[256];
};

static void widen_stat(const struct stat *s, struct stat12 *o) {
    memset(o, 0, sizeof(*o));
    o->st_dev = s->st_dev;
    o->st_ino = s->st_ino;
    o->st_nlink = s->st_nlink;
    o->st_mode = s->st_mode;
    o->st_uid = s->st_uid;
    o->st_gid = s->st_gid;
    o->st_rdev = s->st_rdev;
    o->st_atim = s->st_atim;
    o->st_mtim = s->st_mtim;
    o->st_ctim = s->st_ctim;
    o->st_birthtim = s->st_birthtim;
    o->st_size = s->st_size;
    o->st_blocks = s->st_blocks;
    o->st_blksize = s->st_blksize;
    o->st_flags = s->st_flags;
    o->st_gen = s->st_gen;
}

int __real_stat(const char *, struct stat *);
int __real_lstat(const char *, struct stat *);
int __real_fstat(int, struct stat *);
int __real_fstatat(int, const char *, struct stat *, int);
struct dirent *__real_readdir(DIR *);

#define WIDEN_CALL(call)            \
    struct stat s;                  \
    int r = (call);                 \
    if (r == 0) widen_stat(&s, out); \
    return r;

int __wrap_stat(const char *path, struct stat12 *out) { WIDEN_CALL(__real_stat(path, &s)) }
int __wrap_lstat(const char *path, struct stat12 *out) { WIDEN_CALL(__real_lstat(path, &s)) }
int __wrap_fstat(int fd, struct stat12 *out) { WIDEN_CALL(__real_fstat(fd, &s)) }
int __wrap_fstatat(int fd, const char *path, struct stat12 *out, int flag) {
    WIDEN_CALL(__real_fstatat(fd, path, &s, flag))
}

// One entry per thread: std copies the name out before its next readdir call.
struct dirent12 *__wrap_readdir(DIR *dir) {
    static __thread struct dirent12 out;
    struct dirent *d = __real_readdir(dir);
    if (d == NULL) {
        return NULL;
    }
    memset(&out, 0, offsetof(struct dirent12, d_name));
    out.d_fileno = d->d_fileno;
    out.d_reclen = sizeof(out);
    out.d_type = d->d_type;
    out.d_namlen = d->d_namlen;
    memcpy(out.d_name, d->d_name, d->d_namlen);
    out.d_name[d->d_namlen] = '\0';
    return &out;
}
