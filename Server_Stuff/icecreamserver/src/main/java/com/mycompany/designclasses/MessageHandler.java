
/*
 * Distinguishes this design class from MessageHandler in package
 * com.mycompany.serverutilities.
 */
package com.mycompany.designclasses;

// Imports classes.
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

/**
 * Defines class MessageHandler whose instances serve as message handlers
 * between message interfaces and controllers.
 * @version 0.0
 * @author Tom Lever
 */
public class MessageHandler implements HttpHandler {

    // Commented out to allow building.    
    //private final Controller controller;
    
    /**
     * Defines constructor MessageHandler which sets attributes of this message
     * handler with inputs.
     * @param controllerToUse
     */
    // Commented out to allow building.
    //public MessageHandler(Controller controllerToUse) { }
    
    /**
     * Defines method handle to pick up an HTTP message from a message
     * interface, extracting the ice cream application message in that HTTP
     * message, and handing that ice cream application method to a controller
     * for processing.
     * @param httpExchange
     */
    @Override
    public void handle(HttpExchange httpExchange) { }
    
    /**
     * Defines getIceCreamApplicationMessage, which gets an ice cream
     * application method from an HTTP message associated with an HttpExchange.
     * @param httpExchangeToUse
     * @return 
     */
    private IceCreamApplicationMessage getIceCreamApplicationMessage(
        HttpExchange httpExchangeToUse) {
        // Functionality to get an ice cream application message from an HTTP
        // message associated with an HttpExchange.
        return new IceCreamApplicationMessage();
    }
}