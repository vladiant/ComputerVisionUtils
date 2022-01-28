// https://stackoverflow.com/questions/703316/whats-the-easiest-way-to-display-an-image-in-c-linux

#include <gtk/gtk.h>

void destroy(void) { gtk_main_quit(); }

int main(int argc, char** argv) {
  GtkWidget* window;
  GtkWidget* image;

  gtk_init(&argc, &argv);

  window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  image = gtk_image_new_from_file(argv[1]);

  gtk_signal_connect(GTK_OBJECT(window), "destroy", GTK_SIGNAL_FUNC(destroy),
                     NULL);

  gtk_container_add(GTK_CONTAINER(window), image);

  gtk_widget_show_all(window);

  gtk_main();

  return 0;
}
