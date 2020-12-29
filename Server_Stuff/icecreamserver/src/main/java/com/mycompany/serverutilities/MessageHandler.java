
// Allows executable to find class MessageHandler.
package com.mycompany.serverutilities;

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
    
    private final Controller controller;
    
    /**
     * Defines constructor MessageHandler which sets attributes of this message
     * handler with inputs.
     * @param controllerToUse
     */
    public MessageHandler(Controller controllerToUse) {
        System.out.println("MessageHandler constructor: Started.");
        
        this.controller = controllerToUse;
    }
    
    /**
     * Defines method handle which transfers an HTTP message from a message
     * interface to a controller for processing.
     * @param httpExchange
     */
    @Override
    public void handle(HttpExchange httpExchange) {
        System.out.println("MessageHandler.handle: Started.");
        
        this.controller.process();
        System.out.println(
            "MessageHandler.handle: Called this.controller.process.");
    }
}