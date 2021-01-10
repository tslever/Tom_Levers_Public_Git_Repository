
// Allows executable to find class MessageHandler.
package com.mycompany.serverutilities.clientcommunicationutilities;

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
class MessageHandler implements HttpHandler {
    
    private final static Logger logger =
        Logger.getLogger(MessageHandler.class.getName());
    
    private final Controller controller;
    
    /**
     * Defines constructor MessageHandler which sets attributes of this message
     * handler with inputs.
     * @param controllerToUse
     */
    public MessageHandler(Controller controllerToUse) {
        logger.log(new LogRecord(Level.INFO,
            "MessageHandler constructor: Started."));
        
        this.controller = controllerToUse;
        logger.log(new LogRecord(Level.INFO,
            "MessageHandler constructor: Sets controller as controllerToUse."));
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
        logger.log(new LogRecord(Level.INFO,
            "MessageHandler.handle: Started."));
        
        IceCreamApplicationMessage iceCreamApplicationMessage =
            getIceCreamApplicationMessage(httpExchange);
        logger.log(new LogRecord(Level.INFO,
            "Got ice cream application message from HTTP message from " +
            "client."));
        
        // Must use try block because handle does not throw any exceptions.
        try {
            this.controller.process(iceCreamApplicationMessage);
            logger.log(new LogRecord(Level.INFO,
                "MessageHandler.handle: Called this.controller.process."));
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
            logger.log(new LogRecord(Level.INFO,
                "MessageHandler.getIceCreamApplicationMessage: Started."));
        
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
            logger.log(new LogRecord(Level.INFO,
                "MessageHandler.getIceCreamApplicationMessage: The body of " +
                "the client message is '" + bodyOfClientMessage + "'."));
        }
        catch (IOException e) {
            logger.log(new LogRecord(Level.SEVERE,
                e.toString() + "\n" +
                "Caught IOException thrown by InputStreamReader constructor."));
        }
        
        logger.log(new LogRecord(Level.INFO,
            "MessageHandler.getIceCreamApplicationMessage: Returning new " +
            "ice cream application message based on body of client message."));
        return new IceCreamApplicationMessage(bodyOfClientMessage);
    }
}