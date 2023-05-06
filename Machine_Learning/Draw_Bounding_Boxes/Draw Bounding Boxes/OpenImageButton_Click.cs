
// Allows use of the await keyword.
using System;

// Create namespace Draw_Bounding_Boxes to contain all classes associated with our app.
namespace Draw_Bounding_Boxes
{
    // Create class MainPage that inherits fields and methods from Windows.UI.Xaml.Controls.Page and
    // is used to declare and define user-interface elements and functionality.
    public sealed partial class MainPage : Windows.UI.Xaml.Controls.Page
    {
        // Create OpenImageButton_Click to handle clicking openImageButton.
        private async void OpenImageButton_Click(object sender, Windows.UI.Xaml.RoutedEventArgs e)
        {
            // Open a dialog box to allow opening a JPEG image.
            // The result is either a StorageFile representing the image or null if the user canceled opening.
            Windows.Storage.Pickers.FileOpenPicker pickerToOpenImage = new Windows.Storage.Pickers.FileOpenPicker();
            pickerToOpenImage.SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.PicturesLibrary;
            pickerToOpenImage.FileTypeFilter.Clear();
            pickerToOpenImage.FileTypeFilter.Add(".png");
            pickerToOpenImage.FileTypeFilter.Add(".jpg");
            Windows.Storage.StorageFile fileToOpen = await pickerToOpenImage.PickSingleFileAsync();

            // If an image has been opened...
            if (fileToOpen != null)
            {
                // Refresh imageCanvas.
                this.ImageCanvas.Children.Clear();

                // Create an image object representing the opened image.
                Windows.UI.Xaml.Media.Imaging.BitmapImage bitmapImage = new Windows.UI.Xaml.Media.Imaging.BitmapImage();
                using (Windows.Storage.Streams.IRandomAccessStream fileStream = await fileToOpen.OpenAsync(Windows.Storage.FileAccessMode.Read))
                {
                    bitmapImage.SetSource(fileStream);
                }
                this.ImageToAnalyze = new Windows.UI.Xaml.Controls.Image();
                this.ImageToAnalyze.Source = bitmapImage;

                // Maximize and center imageToAnalyze in app.
                double aspectRatioOfImage = System.Convert.ToDouble(bitmapImage.PixelWidth) / System.Convert.ToDouble(bitmapImage.PixelHeight);
                double aspectRatioOfImageCanvas = this.ImageCanvas.Width / this.ImageCanvas.Height;
                if (aspectRatioOfImage < aspectRatioOfImageCanvas)
                {
                    this.ImageToAnalyze.Height = this.ImageCanvas.Height;
                    this.ImageToAnalyze.Width = this.ImageCanvas.Height * aspectRatioOfImage;
                    Windows.UI.Xaml.Controls.Canvas.SetLeft(this.ImageToAnalyze, (this.ImageCanvas.Width - this.ImageToAnalyze.Width) / 2);
                    Windows.UI.Xaml.Controls.Canvas.SetTop(this.ImageToAnalyze, 0);
                } // if
                else
                {
                    this.ImageToAnalyze.Width = this.ImageCanvas.Width;
                    this.ImageToAnalyze.Height = this.ImageCanvas.Width / aspectRatioOfImage;
                    Windows.UI.Xaml.Controls.Canvas.SetLeft(this.ImageToAnalyze, 0);
                    Windows.UI.Xaml.Controls.Canvas.SetTop(this.ImageToAnalyze, (this.ImageCanvas.Height - this.ImageToAnalyze.Height) / 2);
                } // else
                this.ImageCanvas.Children.Add(this.ImageToAnalyze);

                // Add functionality relating to a mouse pointer moving over imageToAnalyze, a mouse clicking the image, and a mouse pointer passing beyond image boundaries.
                this.PositionOfLeftEdgeOfImageRelativeToLeftEdgeOfImageCanvas = (this.ImageCanvas.Width - this.ImageToAnalyze.Width) / 2;
                this.PositionOfRightEdgeOfImageRelativeToLeftEdgeOfImageCanvas = (this.ImageCanvas.Width + this.ImageToAnalyze.Width) / 2;
                this.PositionOfTopEdgeOfImageRelativeToTopEdgeOfImageCanvas = (this.ImageCanvas.Height - this.ImageToAnalyze.Height) / 2;
                this.PositionOfBottomEdgeOfImageRelativeToTopEdgeOfImageCanvas = (this.ImageCanvas.Height + this.ImageToAnalyze.Height) / 2;

                // Reset clickState and arrayListOfBoundingBoxEncodings.
                this.ClickState = 0;
                this.ArrayListOfBoundingBoxEncodings = new System.Collections.ArrayList();
            } // if
        } // private async void OpenImageButton_Click
    }
}