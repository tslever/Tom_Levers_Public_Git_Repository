
// Allows executable to find class MessageHandler.
package com.mycompany.serverutilities;

// Imports classes.
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.LogRecord;

/**
 * Defines class MessageHandler whose instances serve as message handlers
 * between message interfaces and controllers.
 * @version 0.0
 * @author Tom Lever
 */
public class MessageHandler implements HttpHandler {
    
    private final Controller controller;
    private final static Logger logger =
        Logger.getLogger(MessageHandler.class.getName());
    
    /**
     * Defines constructor MessageHandler which sets attributes of this message
     * handler with inputs.
     * @param controllerToUse
     */
    public MessageHandler(Controller controllerToUse) {
        System.out.println("MessageHandler constructor: Started.");
        
        this.controller = controllerToUse;
        System.out.println(
            "MessageHandler constructor: Sets controller as controllerToUse.");
    }
    
    /**
     * Defines method handle to pick up an HTTP message from a message
     * interface, extract the ice cream application message in that HTTP
     * message, and hand that ice cream application method to a controller
     * for processing.
     * @param httpExchange
     */
    @Override
    public void handle(HttpExchange httpExchange) {
        System.out.println("MessageHandler.handle: Started.");
        
        IceCreamApplicationMessage iceCreamApplicationMessage =
            getIceCreamApplicationMessage(httpExchange);
        System.out.println(
            "Got ice cream application message from HTTP message from client.");
        
        // Must use try block because handle does not throw any exceptions.
        try {
            this.controller.process(iceCreamApplicationMessage);
            System.out.println(
                "MessageHandler.handle: Called this.controller.process.");
        }
        catch(Exception e) {
            logger.log(new LogRecord(Level.SEVERE,
                e.toString() + "\n" +
                "BasicController must be extended; process must be " +
                "overridden."));
        }
        // TODO: Catch BasicControllerMustBeExtendedException instead of
        // Exception.
    }
    
    /**
     * Defines getIceCreamApplicationMessage, which gets an ice cream
     * application method from an HTTP message associated with an HttpExchange.
     * @param httpExchangeToUse
     * @return 
     */
    private IceCreamApplicationMessage getIceCreamApplicationMessage(
        HttpExchange httpExchangeToUse) {
        
        String bodyOfClientMessage = "";
        
        // NetBeans wanted me to use a try-with-resources block.
        // Additionally, try block here simplifies method handle.
        try (
            InputStream inputStream = httpExchangeToUse.getRequestBody();
            InputStreamReader inputStreamReader =
                new InputStreamReader(inputStream, "UTF-8");
            BufferedReader bufferedReader =
                new BufferedReader(inputStreamReader);
        ) {
            StringBuilder stringBuilder = new StringBuilder();
            int readByteAsInt;
            while( (readByteAsInt = bufferedReader.read()) != -1 ) {
                stringBuilder.append( (char)readByteAsInt );
            }

           bodyOfClientMessage = stringBuilder.toString();
           System.out.printf(
                "MessageHandler.getIceCreamApplicationMessage: The body of " +
                "the client message is '%s'.\n", bodyOfClientMessage);
        }
        catch (IOException e) {
            logger.log(new LogRecord(Level.SEVERE,
                e.toString() + "\n" +
                "Caught IOException thrown by InputStreamReader constructor."));
        }
        
        return new IceCreamApplicationMessage(bodyOfClientMessage);
    }
}