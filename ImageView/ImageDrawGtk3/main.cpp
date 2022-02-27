#include <gtk/gtk.h>

int width = 320, height = 240;

GdkPixbuf *pixbuf = gdk_pixbuf_new(GDK_COLORSPACE_RGB, FALSE, 8, width, height);
int rowstride = gdk_pixbuf_get_rowstride(pixbuf);
guchar *pixels = gdk_pixbuf_get_pixels(pixbuf);
GtkImage *image;

inline void drawPixel(int x, int y, char red, char green, char blue) {
  *(pixels + y * rowstride + x * 3) = red;
  *(pixels + y * rowstride + x * 3 + 1) = green;
  *(pixels + y * rowstride + x * 3 + 2) = blue;
}

gboolean game_loop(GtkWidget *widget, GdkFrameClock *clock, gpointer data) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      drawPixel(i, j, rand() % 255, rand() % 255, rand() % 255);
    }
  }
  gtk_image_set_from_pixbuf(image, pixbuf);
  return 1;
}

int main() {
  gtk_init(NULL, NULL);
  GtkWidget *window, *box;
  image = GTK_IMAGE(gtk_image_new());
  window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  gtk_box_pack_start(GTK_BOX(box), GTK_WIDGET(image), TRUE, TRUE, 0);
  gtk_container_add(GTK_CONTAINER(window), box);
  g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
  gtk_widget_add_tick_callback(window, game_loop, NULL, NULL);
  gtk_widget_show_all(window);
  gtk_main();
}