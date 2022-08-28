
namespace Draw_Bounding_Boxes
{
    /// <summary>
    /// Provides application-specific behavior to supplement the default Application class.
    /// </summary>
    sealed partial class App : Windows.UI.Xaml.Application
    {
        /// <summary>
        /// Invoked when the application is launched normally by the end user.  Other entry points
        /// will be used such as when the application is launched to open a specific file.
        /// </summary>
        /// <param name="e">Details about the launch request and process.</param>
        protected override void OnLaunched(Windows.ApplicationModel.Activation.LaunchActivatedEventArgs e)
        {
            // Resize app.
            uint screenWidthInRawPixels = Windows.Graphics.Display.DisplayInformation.GetForCurrentView().ScreenWidthInRawPixels;
            uint screenHeightInRawPixels = Windows.Graphics.Display.DisplayInformation.GetForCurrentView().ScreenHeightInRawPixels;
            double rawPixelsPerViewPixel = Windows.Graphics.Display.DisplayInformation.GetForCurrentView().RawPixelsPerViewPixel;
            double screenWidthInViewPixels = System.Convert.ToDouble(screenWidthInRawPixels) / rawPixelsPerViewPixel;
            double screenHeightInViewPixels = System.Convert.ToDouble(screenHeightInRawPixels) / rawPixelsPerViewPixel;
            
            // If offsetToScreenWidthInViewPixels is less than 15,
            // on first load app will be of default size, and on second load app will be full screen.
            // A loaded image will have height equal to full screen height minus app title bar height minus app toolbar height minus 5 view pixels of padding.
            // Part of a loaded image with aspect ratio less than one will be behind Windows taskbar.
            // This is all very complicated and undesirable.
            // If offsetToScreenHeightInViewPixels is less than 40,
            // on first load app will be of default size, and on second load app will be full screen.
            // A loaded image will have height equal to full screen height minus app title bar height minus app toolbar height minus 5 view pixels of padding.
            // Part of a loaded image with aspect ratio less than one will be behind Windows taskbar.
            // This is all very complicated and undesirable.
            // If offsetToScreenWidthInViewPixels is greater than or equal to 15 and offsetToScreenHeightInViewPixels is greater than or equal to 40,
            // on first load app will be of PreferredLaunchViewSize, and a loaded image with aspect ratio less than one will have height exactly equal to height of app minus app title bar height minus app toolbar height.
            // If PreferredLaunchViewSize.Height is only screenHeightInViewPixels - offsetToScreenHeightInViewPixels,
            // part of app and a loaded image with aspect ratio less than one will be behind taskbar.
            // If taskbarHeight is taken off of screenHeightInViewPixels - offsetToScreenHeightInViewPixels,
            // bottom of app and coincident bottom of loaded image will be slightly above taskbar.
            // I consider this ideal.
            double offsetToScreenWidthInViewPixels = 15;
            double offsetToScreenHeightInViewPixels = 40;
            double taskbarHeight = 40;
            Windows.UI.ViewManagement.ApplicationView.PreferredLaunchViewSize = new Windows.Foundation.Size(screenWidthInViewPixels - offsetToScreenWidthInViewPixels, screenHeightInViewPixels - offsetToScreenHeightInViewPixels - taskbarHeight);
            Windows.UI.ViewManagement.ApplicationView.PreferredLaunchWindowingMode = Windows.UI.ViewManagement.ApplicationViewWindowingMode.PreferredLaunchViewSize;

            // Set the app window to a new Frame.
            Windows.UI.Xaml.Controls.Frame rootFrame = new Windows.UI.Xaml.Controls.Frame();
            Windows.UI.Xaml.Window.Current.Content = rootFrame;

            // Navigate the frame to the initial default page.
            rootFrame.Navigate(typeof(MainPage), e.Arguments);

            // Attempts to activate the application window by bringing it to the foreground and setting the input focus to it.
            Windows.UI.Xaml.Window.Current.Activate();

        } // protected override void OnLaunched
    } // sealed partial class App
} // namespace Draw_Bounding_Boxes
