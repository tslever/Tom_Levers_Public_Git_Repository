
// Allows use of the await keyword.
using System;


// Create namespace Draw_Bounding_Boxes to contain all classes associated with our app.
namespace Draw_Bounding_Boxes
{
    // Create class MainPage that inherits fields and methods from Windows.UI.Xaml.Controls.Page and
    // is used to declare and define user-interface elements and functionality.
    public sealed partial class MainPage : Windows.UI.Xaml.Controls.Page
    {
        // Create SaveLabelFileButton_Click to handle clicking saveLabelFileButton.
        private async void SaveLabelFileButton_Click(object sender, Windows.UI.Xaml.RoutedEventArgs e)
        {
            // Open a dialog box to allow opening a plain-text label file for writing.
            // The result is either a StorageFile representing the label file or null if the user canceled opening.
            Windows.Storage.Pickers.FileSavePicker pickerToSaveLabelFile = new Windows.Storage.Pickers.FileSavePicker();
            pickerToSaveLabelFile.SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.PicturesLibrary;
            pickerToSaveLabelFile.FileTypeChoices.Add("Plain Text", new System.Collections.Generic.List<string>() { ".txt" });
            Windows.Storage.StorageFile fileToWhichToSave = await pickerToSaveLabelFile.PickSaveFileAsync();

            // If fileToWhichToSave has been opened for writing and arrayListOfBoundingBoxEncodings exists and arrayListOfBoundingBoxEncodings contains boundingBoxEncoding's...
            string contentsOfLabelFile = "";
            if ((fileToWhichToSave != null) && (this.ArrayListOfBoundingBoxEncodings != null) && (this.ArrayListOfBoundingBoxEncodings.Count > 0))
            {
                // Add all bounding-box encodings to a string representing the contents of the label file.
                foreach (string[] encoding in this.ArrayListOfBoundingBoxEncodings)
                {
                    contentsOfLabelFile += encoding[0];
                    contentsOfLabelFile += " ";
                    contentsOfLabelFile += encoding[1];
                    contentsOfLabelFile += " ";
                    contentsOfLabelFile += encoding[2];
                    contentsOfLabelFile += " ";
                    contentsOfLabelFile += encoding[3];
                    contentsOfLabelFile += " ";
                    contentsOfLabelFile += encoding[4];
                    contentsOfLabelFile += "\n";
                } // foreach

                // Write contentsOfLabelFile to fileToWhichToSave.
                await Windows.Storage.FileIO.WriteTextAsync(fileToWhichToSave, contentsOfLabelFile);
            } // if
        } // private async void SaveLabelFileButton_Click
    }
}
