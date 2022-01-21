// https://stackoverflow.com/questions/703316/whats-the-easiest-way-to-display-an-image-in-c-linux

#include <gtk/gtk.h>

int width = 320, height = 240;

GtkWidget* image;
GdkPixbuf* pixbuf;
guchar* pixels;
int rowstride;

static gboolean on_timeout(gpointer user_data) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      *(pixels + j * rowstride + i * 3) = rand() % 255;
      *(pixels + j * rowstride + i * 3 + 1) = rand() % 255;
      *(pixels + j * rowstride + i * 3 + 2) = rand() % 255;
    }
  }
  gtk_image_set_from_pixbuf(GTK_IMAGE(image), pixbuf);

  return G_SOURCE_CONTINUE; /* or G_SOURCE_REMOVE when you want to stop */
}

void destroy(void) { gtk_main_quit(); }

int main(int argc, char** argv) {
  GtkWidget* window;

  gtk_init(&argc, &argv);

  window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

  image = gtk_image_new();

  pixbuf = gdk_pixbuf_new(GDK_COLORSPACE_RGB, FALSE, 8, width, height);
  rowstride = gdk_pixbuf_get_rowstride(pixbuf);
  pixels = gdk_pixbuf_get_pixels(pixbuf);

  gtk_signal_connect(GTK_OBJECT(window), "destroy", GTK_SIGNAL_FUNC(destroy),
                     NULL);

  gtk_container_add(GTK_CONTAINER(window), image);

  g_timeout_add(25 /* milliseconds */, on_timeout, NULL);

  gtk_widget_show_all(window);

  gtk_main();

  return 0;
}
