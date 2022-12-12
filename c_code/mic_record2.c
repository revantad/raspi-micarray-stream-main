#define ALSA_PCM_NEW_HW_PARAMS_API
#include <termios.h>
#include <alsa/asoundlib.h>

struct termios stdin_orig;  // Structure to save parameters

void term_reset() {
        tcsetattr(STDIN_FILENO,TCSANOW,&stdin_orig);
        tcsetattr(STDIN_FILENO,TCSAFLUSH,&stdin_orig);
}

void term_nonblocking() {
        struct termios newt;
        tcgetattr(STDIN_FILENO, &stdin_orig);
        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK); // non-blocking
        newt = stdin_orig;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);

        atexit(term_reset);
}

int main() {
  int key=0;
  long loops;
  int rc;
  int size;
  snd_pcm_t *handle;
  snd_pcm_hw_params_t *params;
  unsigned int val;
  int dir;
  snd_pcm_uframes_t frames;
  char *buffer;

  /* Open PCM device for recording (capture). */
  rc = snd_pcm_open(&handle, "default", SND_PCM_STREAM_CAPTURE, 0);
  if (rc < 0) {
    fprintf(stderr, "unable to open pcm device: %s\n", snd_strerror(rc));
    exit(1);
  }

  /* Allocate a hardware parameters object. */
  snd_pcm_hw_params_alloca(&params);

  /* Fill it in with default values. */
  snd_pcm_hw_params_any(handle, params);

  /* Set the desired hardware parameters. */

  /* Interleaved mode */
  snd_pcm_hw_params_set_access(handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);

  /* Signed 16-bit little-endian format */
  snd_pcm_hw_params_set_format(handle, params, SND_PCM_FORMAT_S16_LE);

  /* One channel (mono) */
  snd_pcm_hw_params_set_channels(handle, params, 1);

  /* 16000 bits/second sampling rate */
  val = 16000;
  snd_pcm_hw_params_set_rate_near(handle, params, &val, &dir);

  /* Set period size to 2048 frames. */
  frames = 2048;
  snd_pcm_hw_params_set_period_size_near(handle, params, &frames, &dir);

  /* Write the parameters to the driver */
  rc = snd_pcm_hw_params(handle, params);
  if (rc < 0) {
    fprintf(stderr, "unable to set hw parameters: %s\n", snd_strerror(rc));
    exit(1);
  }

  /* Use a buffer large enough to hold one period */
  snd_pcm_hw_params_get_period_size(params, &frames, &dir);
  size = frames * 2; /* 2 bytes/sample, 1 channels */
  buffer = (char *) malloc(size);

  while (key == 0) 
  {

    rc = snd_pcm_readi(handle, buffer, frames);
    if (rc == -EPIPE) 
    {
      /* EPIPE means overrun */
      fprintf(stderr, "overrun occurred\n");
      snd_pcm_prepare(handle);
    } 
    else if (rc < 0)
    {
      fprintf(stderr, "error from read: %s\n", snd_strerror(rc));
    } 
    else if (rc != (int)frames) 
    {
      fprintf(stderr, "short read, read %d frames\n", rc);
    }

    rc = write(1, buffer, size);

    if (rc != size)
      fprintf(stderr, "short write: wrote %d bytes\n", rc);
    key = getchar();
  }

  snd_pcm_drain(handle);
  snd_pcm_close(handle);
  free(buffer);

  return 0;
}