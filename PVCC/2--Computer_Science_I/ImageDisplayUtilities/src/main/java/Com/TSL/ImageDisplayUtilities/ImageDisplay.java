package Com.TSL.ImageDisplayUtilities;

import javafx.application.Application;
import javafx.geometry.Rectangle2D;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

//************************************************************************
//  ImageDisplay.java       Author: Lewis/Loftus
//
//  Demonstrates a the use of Image and ImageView objects.
//************************************************************************

public class ImageDisplay extends Application
{
	private static String pathToImage;
	
    //--------------------------------------------------------------------
    //  Displays an image centered in a window.
    //--------------------------------------------------------------------
    public void start(Stage primaryStage)
    {
        Image img = new Image(pathToImage);
        ImageView imgView = new ImageView(img);

        StackPane pane = new StackPane(imgView);
        pane.setStyle("-fx-background-color: cornsilk");
        imgView.setViewport(new Rectangle2D(200, 80, 70, 60));
        
        Scene scene = new Scene(pane, 500, 350);

        primaryStage.setTitle("Image Display");
        primaryStage.setScene(scene);
        primaryStage.show();
    }
    
    public static void main(String[] args)
    {
    	pathToImage = args[0];
        launch(args);
    }
}