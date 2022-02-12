// https://tipsfordev.com/load-gif-into-gtk-3-window-c-c
// https://stackoverflow.com/questions/39574472/gtk-3-where-and-how-do-i-specify-the-contents-of-the-main-loop

#include <gtk/gtk.h>
#include <stdio.h>
#include <stdlib.h>

GdkPixbuf *load_pixbuf_from_file(const char *filename) {
  GError *error = NULL;
  GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file(filename, &error);

  if (pixbuf == NULL) {
    g_print("Error loading file: %d : %s\n", error->code, error->message);
    g_error_free(error);
    exit(1);
  }
  return pixbuf;
}

int main(int argc, char **argv) {
  GtkWidget *window = NULL;
  GdkPixbuf *image = NULL;
  GdkPixbufAnimation *anim = NULL;
  GtkWidget *widget = NULL;

  gtk_init(&argc, &argv);

  image = load_pixbuf_from_file(argv[1]);
  widget = gtk_image_new_from_pixbuf(image);

  window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  gtk_window_set_title(GTK_WINDOW(window), "Load Image");
  gtk_window_set_default_size(GTK_WINDOW(window), 400, 300);
  gtk_window_set_transient_for(GTK_WINDOW(window), NULL);

  GtkWidget *hbox = NULL;

  hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, FALSE);
  gtk_container_add(GTK_CONTAINER(window), hbox);

  // Button
  GtkWidget *button = NULL;
  button = gtk_button_new_with_label("Button");
  gtk_box_pack_start(GTK_BOX(hbox), button, FALSE, FALSE, 0);

  // Image
  gtk_box_pack_start(GTK_BOX(hbox), widget, FALSE, FALSE, 0);

  g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

  gtk_widget_show_all(window);
  gtk_main();
  return 0;
}