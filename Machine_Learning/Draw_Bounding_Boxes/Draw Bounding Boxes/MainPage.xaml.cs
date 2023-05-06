
// Create namespace Draw_Bounding_Boxes to contain all classes associated with our app.
namespace Draw_Bounding_Boxes
{
    // Create class MainPage that inherits fields and methods from Windows.UI.Xaml.Controls.Page and
    // is used to declare and define user-interface elements and functionality.
    public sealed partial class MainPage : Windows.UI.Xaml.Controls.Page
    {
        // Defined in constructor public MainPage.
        // Used in private void TextBoxForClassIndex_TextChanged, private Windows.UI.Xaml.Media.SolidColorBrush GetColor.
        private Windows.UI.Xaml.Controls.TextBox TextBoxForClassIndex { get; set; }

        // Defined in constructor public MainPage.
        // Used in private void OpenImageButton_Click, private void SaveLabeledImageButton_Click, private void ImageCanvas_PointerMoved, private void ImageCanvas_PointerPressed, private void ImageCanvas_PointerExited.
        private Windows.UI.Xaml.Controls.Canvas ImageCanvas { get; set; }

        // Defined in OpenImageButton_Click.
        // Used in SaveLabelFileButton_Click, ImageCanvas_PointerPressed.
        private System.Collections.ArrayList ArrayListOfBoundingBoxEncodings { get; set; }

        // Defined in private void OpenImageButton_Click.
        // Used in ImageCanvas_PointerPressed.
        private Windows.UI.Xaml.Controls.Image ImageToAnalyze { get; set; }

        // Defined in OpenImageButton_Click.
        // Used in ImageCanvas_PointerMoved, ImageCanvas_PointerPressed.
        private double PositionOfLeftEdgeOfImageRelativeToLeftEdgeOfImageCanvas { get; set; }
        private double PositionOfRightEdgeOfImageRelativeToLeftEdgeOfImageCanvas { get; set; }
        private double PositionOfTopEdgeOfImageRelativeToTopEdgeOfImageCanvas { get; set; }
        private double PositionOfBottomEdgeOfImageRelativeToTopEdgeOfImageCanvas { get; set; }

        // Defined in OpenImageButton_Click.
        // Used in ImageCanvas_PointerMoved, ImageCanvas_PointerPressed.
        private uint ClickState { get; set; }

        // Defined in ImageCanvas_PointerPressed.
        // Used in ImageCanvas_PointerMoved.
        private double PositionOfFirstClickRelativeToLeftEdgeOfImageCanvas { get; set; }
        private double PositionOfFirstClickRelativeToTopEdgeOfImageCanvas { get; set; }


        // Create constructor public MainPage.
        public MainPage()
        {
            // Necessary to instantiate this Page, add a stackPanel to this Page, et cetera.
            this.InitializeComponent();

            // Find width of app in view pixels and height between bottom of app and bottom of title bar in view pixels.
            double widthOfAppInViewPixels = Windows.UI.ViewManagement.ApplicationView.PreferredLaunchViewSize.Width;
            double heightBetweenBottomOfAppAndBottomOfTitleBarInViewPixels = Windows.UI.ViewManagement.ApplicationView.PreferredLaunchViewSize.Height;

            // Create a stackPanel.
            Windows.UI.Xaml.Controls.StackPanel stackPanel = new Windows.UI.Xaml.Controls.StackPanel();

            // Create a toolbar with width equal to the width of the app, height equal to 50 view pixels, and background color of light blue that has one row and three columns.
            Windows.UI.Xaml.Controls.Grid toolbar = new Windows.UI.Xaml.Controls.Grid();
            toolbar.Width = widthOfAppInViewPixels;
            toolbar.Height = 50;
            toolbar.Background = new Windows.UI.Xaml.Media.SolidColorBrush(Windows.UI.Colors.AliceBlue);
            Windows.UI.Xaml.Controls.RowDefinition row = new Windows.UI.Xaml.Controls.RowDefinition();
            toolbar.RowDefinitions.Add(row);
            Windows.UI.Xaml.Controls.ColumnDefinition columnForOpenImageButton = new Windows.UI.Xaml.Controls.ColumnDefinition();
            Windows.UI.Xaml.Controls.ColumnDefinition columnForLoadBoundingBoxesButton = new Windows.UI.Xaml.Controls.ColumnDefinition();
            Windows.UI.Xaml.Controls.ColumnDefinition columnForTextboxForClassIndex = new Windows.UI.Xaml.Controls.ColumnDefinition();
            Windows.UI.Xaml.Controls.ColumnDefinition columnForSaveLabeledImageButton = new Windows.UI.Xaml.Controls.ColumnDefinition();
            Windows.UI.Xaml.Controls.ColumnDefinition columnForSaveLabelFileButton = new Windows.UI.Xaml.Controls.ColumnDefinition();
            columnForOpenImageButton.Width = new Windows.UI.Xaml.GridLength(widthOfAppInViewPixels * 3 / 7 / 2);
            columnForLoadBoundingBoxesButton.Width = new Windows.UI.Xaml.GridLength(widthOfAppInViewPixels * 3 / 7 / 2);
            columnForTextboxForClassIndex.Width = new Windows.UI.Xaml.GridLength(widthOfAppInViewPixels / 7);
            columnForSaveLabeledImageButton.Width = new Windows.UI.Xaml.GridLength(widthOfAppInViewPixels * 3 / 7 / 2);
            columnForSaveLabelFileButton.Width = new Windows.UI.Xaml.GridLength(widthOfAppInViewPixels * 3 / 7 / 2);
            toolbar.ColumnDefinitions.Add(columnForOpenImageButton);
            toolbar.ColumnDefinitions.Add(columnForLoadBoundingBoxesButton);
            toolbar.ColumnDefinitions.Add(columnForTextboxForClassIndex);
            toolbar.ColumnDefinitions.Add(columnForSaveLabeledImageButton);
            toolbar.ColumnDefinitions.Add(columnForSaveLabelFileButton);

            // Add to toolbar's columnForOpenImageButton an "Open Image" button.
            Windows.UI.Xaml.Controls.Button openImageButton = new Windows.UI.Xaml.Controls.Button();
            openImageButton.Content = "Open Image";
            openImageButton.Height = 40;
            Windows.UI.Xaml.Controls.Grid.SetRow(openImageButton, 0);
            Windows.UI.Xaml.Controls.Grid.SetColumn(openImageButton, 0);
            openImageButton.HorizontalAlignment = Windows.UI.Xaml.HorizontalAlignment.Center;
            openImageButton.Click += OpenImageButton_Click;
            toolbar.Children.Add(openImageButton);

            // Add to the toolbar's columnForLoadBoundingBoxesButton a "Load Bounding Boxes" button.
            Windows.UI.Xaml.Controls.Button loadBoundingBoxesButton = new Windows.UI.Xaml.Controls.Button();
            loadBoundingBoxesButton.Content = "Load Bounding Boxes";
            loadBoundingBoxesButton.Height = 40;
            Windows.UI.Xaml.Controls.Grid.SetRow(loadBoundingBoxesButton, 0);
            Windows.UI.Xaml.Controls.Grid.SetColumn(loadBoundingBoxesButton, 1);
            loadBoundingBoxesButton.HorizontalAlignment = Windows.UI.Xaml.HorizontalAlignment.Center;
            loadBoundingBoxesButton.Click += LoadBoundingBoxesButton_Click;
            toolbar.Children.Add(loadBoundingBoxesButton);

            // Add to toolbar's columnForTextboxForClassIndex a labeled text box to store a user's choice of class index for an object they are about to bound.
            Windows.UI.Xaml.Controls.TextBlock textblockForClassIndex = new Windows.UI.Xaml.Controls.TextBlock();
            textblockForClassIndex.Text = "Class Index: ";
            textblockForClassIndex.Height = 20;
            Windows.UI.Xaml.Controls.Grid.SetRow(textblockForClassIndex, 0);
            Windows.UI.Xaml.Controls.Grid.SetColumn(textblockForClassIndex, 2);
            toolbar.Children.Add(textblockForClassIndex);
            this.TextBoxForClassIndex = new Windows.UI.Xaml.Controls.TextBox();
            this.TextBoxForClassIndex.Text = "0";
            this.TextBoxForClassIndex.Width = 40;
            this.TextBoxForClassIndex.Height = 40;
            Windows.UI.Xaml.Controls.Grid.SetRow(this.TextBoxForClassIndex, 0);
            Windows.UI.Xaml.Controls.Grid.SetColumn(this.TextBoxForClassIndex, 2);
            this.TextBoxForClassIndex.TextChanged += TextBoxForClassIndex_TextChanged;
            toolbar.Children.Add(this.TextBoxForClassIndex);

            // Add to toolbar's columnForSaveLabeledImageButton a "Save Labeled Image" button.
            Windows.UI.Xaml.Controls.Button saveLabeledImageButton = new Windows.UI.Xaml.Controls.Button();
            saveLabeledImageButton.Content = "Save Labeled Image";
            saveLabeledImageButton.Height = 40;
            Windows.UI.Xaml.Controls.Grid.SetRow(saveLabeledImageButton, 0);
            Windows.UI.Xaml.Controls.Grid.SetColumn(saveLabeledImageButton, 3);
            saveLabeledImageButton.HorizontalAlignment = Windows.UI.Xaml.HorizontalAlignment.Center;
            saveLabeledImageButton.Click += SaveLabeledImageButton_Click;
            toolbar.Children.Add(saveLabeledImageButton);

            // Add to toolbar's columnForSaveLabelFileButton a "Save Label File" button.
            Windows.UI.Xaml.Controls.Button saveLabelFileButton = new Windows.UI.Xaml.Controls.Button();
            saveLabelFileButton.Content = "Save Label File";
            saveLabelFileButton.Height = 40;
            Windows.UI.Xaml.Controls.Grid.SetRow(saveLabelFileButton, 0);
            Windows.UI.Xaml.Controls.Grid.SetColumn(saveLabelFileButton, 4);
            saveLabelFileButton.HorizontalAlignment = Windows.UI.Xaml.HorizontalAlignment.Center;
            saveLabelFileButton.Click += SaveLabelFileButton_Click;
            toolbar.Children.Add(saveLabelFileButton);

            // Add grid to the top of our stackPanel.
            stackPanel.Children.Add(toolbar);

            this.ImageCanvas = new Windows.UI.Xaml.Controls.Canvas();
            this.ImageCanvas.Width = widthOfAppInViewPixels;
            this.ImageCanvas.Height = heightBetweenBottomOfAppAndBottomOfTitleBarInViewPixels - toolbar.Height;
            this.ImageCanvas.PointerMoved += new Windows.UI.Xaml.Input.PointerEventHandler(ImageCanvas_PointerMoved);
            this.ImageCanvas.PointerPressed += new Windows.UI.Xaml.Input.PointerEventHandler(ImageCanvas_PointerPressed);
            this.ImageCanvas.PointerExited += new Windows.UI.Xaml.Input.PointerEventHandler(ImageCanvas_PointerExited);
            stackPanel.Children.Add(this.ImageCanvas);

            // Add stackPanel to the page defined in MainPage.xaml.
            page.Content = stackPanel;

        } // public MainPage
    } // public sealed partial class MainPage
} // namespace Draw_Bounding_Boxes
