
// Allows use of the await keyword.
using System;

// Allow use of "byte[] bytes = pixels.ToArray();".
using System.Runtime.InteropServices.WindowsRuntime;


// Create namespace Draw_Bounding_Boxes to contain all classes associated with our app.
namespace Draw_Bounding_Boxes
{
    // Create class MainPage that inherits fields and methods from Windows.UI.Xaml.Controls.Page and
    // is used to declare and define user-interface elements and functionality.
    public sealed partial class MainPage : Windows.UI.Xaml.Controls.Page
    {
        // Create SaveLabeledImageButton_Click to handle clicking saveLabeledImageButton.
        // Allow use of "byte[] bytes = pixels.ToArray();".
        //using System.Runtime.InteropServices.WindowsRuntime;
        private async void SaveLabeledImageButton_Click(object sender, Windows.UI.Xaml.RoutedEventArgs e)
        {
            Windows.UI.Xaml.Media.Imaging.RenderTargetBitmap renderTargetBitmap = new Windows.UI.Xaml.Media.Imaging.RenderTargetBitmap();
            await renderTargetBitmap.RenderAsync(this.ImageCanvas);

            Windows.Storage.Pickers.FileSavePicker pickerToSaveLabeledImage = new Windows.Storage.Pickers.FileSavePicker();
            pickerToSaveLabeledImage.FileTypeChoices.Add("PNG Image", new string[] { ".png" });
            pickerToSaveLabeledImage.FileTypeChoices.Add("JPEG Image", new string[] { ".jpg" });
            Windows.Storage.StorageFile fileToWhichToSave = await pickerToSaveLabeledImage.PickSaveFileAsync();

            if (fileToWhichToSave != null)
            {
                Windows.Storage.Streams.IBuffer pixels = await renderTargetBitmap.GetPixelsAsync();

                using (Windows.Storage.Streams.IRandomAccessStream stream = await fileToWhichToSave.OpenAsync(Windows.Storage.FileAccessMode.ReadWrite))
                {
                    Windows.Graphics.Imaging.BitmapEncoder encoder = await Windows.Graphics.Imaging.BitmapEncoder.CreateAsync(Windows.Graphics.Imaging.BitmapEncoder.JpegEncoderId, stream);
                    encoder.SetPixelData(
                        Windows.Graphics.Imaging.BitmapPixelFormat.Bgra8,
                        Windows.Graphics.Imaging.BitmapAlphaMode.Ignore,
                        (uint)renderTargetBitmap.PixelWidth,
                        (uint)renderTargetBitmap.PixelHeight,
                        Windows.Graphics.Display.DisplayInformation.GetForCurrentView().LogicalDpi,
                        Windows.Graphics.Display.DisplayInformation.GetForCurrentView().LogicalDpi,
                        pixels.ToArray());
                    await encoder.FlushAsync();
                }
            }
        }
    }
}