
// Create namespace Draw_Bounding_Boxes to contain all classes associated with our app.
namespace Draw_Bounding_Boxes
{
    // Create class MainPage that inherits fields and methods from Windows.UI.Xaml.Controls.Page and
    // is used to declare and define user-interface elements and functionality.
    public sealed partial class MainPage : Windows.UI.Xaml.Controls.Page
    {
        // Create TextBoxForClassIndex_TextChanged to handle changing the text of textBoxForClassIndex.
        private void TextBoxForClassIndex_TextChanged(object sender, Windows.UI.Xaml.Controls.TextChangedEventArgs e)
        {
            // If the text of textBoxForClassIndex is not an integer or is negative or is greater than nine, reset the text to the string equivalent of zero.
            int integerEquivalent;
            if (!int.TryParse(this.TextBoxForClassIndex.Text, out integerEquivalent) || (integerEquivalent < 0) || (integerEquivalent > 19))
            {
                this.TextBoxForClassIndex.Text = "0";
            } // if
        } // private void TextBoxForClassIndex_TextChanged
    }
}