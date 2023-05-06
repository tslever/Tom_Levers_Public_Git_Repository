// Create namespace Draw_Bounding_Boxes to contain all classes associated with our app.
namespace Draw_Bounding_Boxes
{
    // Create class MainPage that inherits fields and methods from Windows.UI.Xaml.Controls.Page and
    // is used to declare and define user-interface elements and functionality.
    public sealed partial class MainPage : Windows.UI.Xaml.Controls.Page
    {
        // Create a method to return a SolidColorBrush based on a class index.
        private Windows.UI.Xaml.Media.SolidColorBrush GetColor(string classIndex)
        {
            switch (classIndex)
            {
                case "0":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.AliceBlue);
                case "1":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.AntiqueWhite);
                case "2":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Aqua);
                case "3":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Aquamarine);
                case "4":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Azure);
                case "5":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Beige);
                case "6":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Bisque);
                case "7":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Black);
                case "8":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.BlanchedAlmond);
                case "9":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Blue);
                case "10":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.BlueViolet);
                case "11":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Brown);
                case "12":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.BurlyWood);
                case "13":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.CadetBlue);
                case "14":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Chartreuse);
                case "15":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Chocolate);
                case "16":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Coral);
                case "17":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.CornflowerBlue);
                case "18":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Cornsilk);
                case "19":
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Crimson);
                default:
                    return new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.Cyan);
            } // switch
        } // private Windows.UI.Xaml.Media.SolidColorBrush GetColor
    }
}
