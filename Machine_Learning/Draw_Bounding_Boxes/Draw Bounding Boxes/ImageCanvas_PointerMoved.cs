// Create namespace Draw_Bounding_Boxes to contain all classes associated with our app.
namespace Draw_Bounding_Boxes
{
    // Create class MainPage that inherits fields and methods from Windows.UI.Xaml.Controls.Page and
    // is used to declare and define user-interface elements and functionality.
    public sealed partial class MainPage : Windows.UI.Xaml.Controls.Page
    {
        // Create ImageCanvas_PointerMoved to handle moving a mouse pointer across an opened image.
        private void ImageCanvas_PointerMoved(object sender, Windows.UI.Xaml.Input.PointerRoutedEventArgs e)
        {
            Windows.Foundation.Point positionOfPointer = e.GetCurrentPoint(this.ImageCanvas).Position;
            double positionOfPointerRelativeToLeftEdgeOfImageCanvas = positionOfPointer.X;
            double positionOfPointerRelativeToTopEdgeOfImageCanvas = positionOfPointer.Y;

            // If the image has not been clicked to draw a bounding box, or if a bounding box has been drawn and clickState reset...
            if (this.ClickState == 0)
            {
                // Remove any horizontal and vertical lines through the mouse pointer.
                this.ImageCanvas.Children.Remove((Windows.UI.Xaml.UIElement)this.ImageCanvas.FindName("horizontalPointerLine"));
                this.ImageCanvas.Children.Remove((Windows.UI.Xaml.UIElement)this.ImageCanvas.FindName("verticalPointerLine"));

                // Draw a new horizontal pointer line.
                Windows.UI.Xaml.Shapes.Line horizontalPointerLine = new Windows.UI.Xaml.Shapes.Line();
                horizontalPointerLine.Name = "horizontalPointerLine";
                horizontalPointerLine.X1 = this.PositionOfLeftEdgeOfImageRelativeToLeftEdgeOfImageCanvas;
                horizontalPointerLine.X2 = this.PositionOfRightEdgeOfImageRelativeToLeftEdgeOfImageCanvas;
                horizontalPointerLine.Y1 = positionOfPointerRelativeToTopEdgeOfImageCanvas;
                horizontalPointerLine.Y2 = positionOfPointerRelativeToTopEdgeOfImageCanvas;
                horizontalPointerLine.Stroke = new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Black);
                horizontalPointerLine.StrokeThickness = 1;
                this.ImageCanvas.Children.Add(horizontalPointerLine);

                // Draw a new vertical pointer line.
                Windows.UI.Xaml.Shapes.Line verticalPointerLine = new Windows.UI.Xaml.Shapes.Line();
                verticalPointerLine.Name = "verticalPointerLine";
                verticalPointerLine.X1 = positionOfPointerRelativeToLeftEdgeOfImageCanvas;
                verticalPointerLine.X2 = positionOfPointerRelativeToLeftEdgeOfImageCanvas;
                verticalPointerLine.Y1 = this.PositionOfTopEdgeOfImageRelativeToTopEdgeOfImageCanvas;
                verticalPointerLine.Y2 = this.PositionOfBottomEdgeOfImageRelativeToTopEdgeOfImageCanvas;
                verticalPointerLine.Stroke = new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Black);
                verticalPointerLine.StrokeThickness = 1;
                this.ImageCanvas.Children.Add(verticalPointerLine);
            } // if

            // If the image has been clicked once, or if a bounding box has been drawn, clickState reset, and the image clicked again...
            if (this.ClickState == 1)
            {
                // Remove any rectangle from the imageCanvas.
                this.ImageCanvas.Children.Remove((Windows.UI.Xaml.UIElement)this.ImageCanvas.FindName("rectangle"));

                // Add a new rectangle named rectangle between the first click point and the present location of a mouse pointer. 
                Windows.UI.Xaml.Shapes.Rectangle rectangle = new Windows.UI.Xaml.Shapes.Rectangle();
                rectangle.Name = "rectangle";
                rectangle.Fill = this.GetColor(this.TextBoxForClassIndex.Text);
                rectangle.Opacity = 0.1;
                rectangle.Width = System.Math.Abs(positionOfPointerRelativeToLeftEdgeOfImageCanvas - this.PositionOfFirstClickRelativeToLeftEdgeOfImageCanvas);
                rectangle.Height = System.Math.Abs(positionOfPointerRelativeToTopEdgeOfImageCanvas - this.PositionOfFirstClickRelativeToTopEdgeOfImageCanvas);
                Windows.UI.Xaml.Controls.Canvas.SetLeft(rectangle, System.Math.Min(positionOfPointerRelativeToLeftEdgeOfImageCanvas, this.PositionOfFirstClickRelativeToLeftEdgeOfImageCanvas));
                Windows.UI.Xaml.Controls.Canvas.SetTop(rectangle, System.Math.Min(positionOfPointerRelativeToTopEdgeOfImageCanvas, this.PositionOfFirstClickRelativeToTopEdgeOfImageCanvas));
                this.ImageCanvas.Children.Add(rectangle);
            } // if
        } // private void ImageCanvas_PointerMoved
    }
}
