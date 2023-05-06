
// Allows use of the await keyword.
using System;

// Allow use of "storageFile.OpenStreamForReadAsync()".
using System.IO;


// Create namespace Draw_Bounding_Boxes to contain all classes associated with our app.
namespace Draw_Bounding_Boxes
{
    // Create class MainPage that inherits fields and methods from Windows.UI.Xaml.Controls.Page and
    // is used to declare and define user-interface elements and functionality.
    public sealed partial class MainPage : Windows.UI.Xaml.Controls.Page
    {
        private async void LoadBoundingBoxesButton_Click(object sender, Windows.UI.Xaml.RoutedEventArgs e)
        {
            Windows.Storage.Pickers.FileOpenPicker picker = new Windows.Storage.Pickers.FileOpenPicker();
            picker.ViewMode = Windows.Storage.Pickers.PickerViewMode.List;
            picker.FileTypeFilter.Add(".txt");
            Windows.Storage.StorageFile storageFile = await picker.PickSingleFileAsync();
            System.IO.Stream stream = await storageFile.OpenStreamForReadAsync();

            using (System.IO.StreamReader streamReader = new System.IO.StreamReader(stream)) {
                while (!streamReader.EndOfStream) {
                    string line = streamReader.ReadLine();
                    string[] values = line.Split(' ');

                    Windows.UI.Xaml.Shapes.Rectangle rectangle = new Windows.UI.Xaml.Shapes.Rectangle();
                    rectangle.Fill = this.GetColor(values[0]);
                    rectangle.Opacity = 0.1;
                    rectangle.Width = System.Convert.ToDouble(values[3]) * this.ImageToAnalyze.Width;
                    rectangle.Height = System.Convert.ToDouble(values[4]) * this.ImageToAnalyze.Height;
                    Windows.UI.Xaml.Controls.Canvas.SetLeft(rectangle, this.PositionOfLeftEdgeOfImageRelativeToLeftEdgeOfImageCanvas + (System.Convert.ToDouble(values[1]) * this.ImageToAnalyze.Width) - (rectangle.Width / 2));
                    Windows.UI.Xaml.Controls.Canvas.SetTop(rectangle, this.PositionOfTopEdgeOfImageRelativeToTopEdgeOfImageCanvas + (System.Convert.ToDouble(values[2]) * this.ImageToAnalyze.Height) - (rectangle.Height / 2));
                    this.ImageCanvas.Children.Add(rectangle);
                }
            }
        }
    }
}
