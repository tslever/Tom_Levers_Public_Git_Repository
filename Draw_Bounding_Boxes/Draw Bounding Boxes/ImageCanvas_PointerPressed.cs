// Create namespace Draw_Bounding_Boxes to contain all classes associated with our app.
namespace Draw_Bounding_Boxes
{
    // Create class MainPage that inherits fields and methods from Windows.UI.Xaml.Controls.Page and
    // is used to declare and define user-interface elements and functionality.
    public sealed partial class MainPage : Windows.UI.Xaml.Controls.Page
    {
        // Create ImageCanvas_PointerPressed to handle clicking an opened image.
        private void ImageCanvas_PointerPressed(object sender, Windows.UI.Xaml.Input.PointerRoutedEventArgs e)
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

                this.PositionOfFirstClickRelativeToLeftEdgeOfImageCanvas = positionOfPointer.X;
                this.PositionOfFirstClickRelativeToTopEdgeOfImageCanvas = positionOfPointer.Y;
                this.ClickState = 1;
                return;
            } // if

            // If the image has been clicked once, or if a bounding box has been drawn, clickState reset, and the image clicked again...
            if (this.ClickState == 1)
            {
                // If a rectangle named rectangle has been drawn...
                if (this.ImageCanvas.FindName("rectangle") != null)
                {
                    // Remove this rectangle from the imageCanvas.
                    this.ImageCanvas.Children.Remove((Windows.UI.Xaml.UIElement)this.ImageCanvas.FindName("rectangle"));

                    // Add a new rectangle between the first click point and the present location of a mouse pointer. 
                    Windows.UI.Xaml.Shapes.Rectangle rectangle = new Windows.UI.Xaml.Shapes.Rectangle();
                    rectangle.Fill = this.GetColor(this.TextBoxForClassIndex.Text);
                    rectangle.Opacity = 0.1; 
                    rectangle.Width = System.Math.Abs(positionOfPointerRelativeToLeftEdgeOfImageCanvas - this.PositionOfFirstClickRelativeToLeftEdgeOfImageCanvas);
                    rectangle.Height = System.Math.Abs(positionOfPointerRelativeToTopEdgeOfImageCanvas - this.PositionOfFirstClickRelativeToTopEdgeOfImageCanvas);
                    Windows.UI.Xaml.Controls.Canvas.SetLeft(rectangle, System.Math.Min(this.PositionOfFirstClickRelativeToLeftEdgeOfImageCanvas, positionOfPointerRelativeToLeftEdgeOfImageCanvas));
                    Windows.UI.Xaml.Controls.Canvas.SetTop(rectangle, System.Math.Min(this.PositionOfFirstClickRelativeToTopEdgeOfImageCanvas, positionOfPointerRelativeToTopEdgeOfImageCanvas));
                    this.ImageCanvas.Children.Add(rectangle);

                    // Add the present rectangle's encoding into the ArrayListOfBoundingBoxEncodings.
                    string[] boundingBoxEncoding = new string[5];
                    boundingBoxEncoding[0] = this.TextBoxForClassIndex.Text;
                    boundingBoxEncoding[1] = System.Convert.ToString(((2 * rectangle.ActualOffset.X + rectangle.Width) / 2 - this.PositionOfLeftEdgeOfImageRelativeToLeftEdgeOfImageCanvas) / this.ImageToAnalyze.Width);
                    boundingBoxEncoding[2] = System.Convert.ToString(((2 * rectangle.ActualOffset.Y + rectangle.Height) / 2 - this.PositionOfTopEdgeOfImageRelativeToTopEdgeOfImageCanvas) / this.ImageToAnalyze.Height);
                    boundingBoxEncoding[3] = System.Convert.ToString(rectangle.Width / this.ImageToAnalyze.Width);
                    boundingBoxEncoding[4] = System.Convert.ToString(rectangle.Height / this.ImageToAnalyze.Height);
                    this.ArrayListOfBoundingBoxEncodings.Add(boundingBoxEncoding);

                    this.ClickState = 0;
                } // if
            } // if
        } // private void ImageCanvas_PointerPressed
    }
}
