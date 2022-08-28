// Create namespace Draw_Bounding_Boxes to contain all classes associated with our app.
namespace Draw_Bounding_Boxes
{
    // Create class MainPage that inherits fields and methods from Windows.UI.Xaml.Controls.Page and
    // is used to declare and define user-interface elements and functionality.
    public sealed partial class MainPage : Windows.UI.Xaml.Controls.Page
    {
        private void ImageCanvas_PointerExited(object sender, Windows.UI.Xaml.Input.PointerRoutedEventArgs e)
        {
            // Remove any horizontal and vertical lines through the mouse pointer.
            this.ImageCanvas.Children.Remove((Windows.UI.Xaml.UIElement)this.ImageCanvas.FindName("horizontalPointerLine"));
            this.ImageCanvas.Children.Remove((Windows.UI.Xaml.UIElement)this.ImageCanvas.FindName("verticalPointerLine"));
        } // private void ImageCanvas_PointerExited
    }
}
