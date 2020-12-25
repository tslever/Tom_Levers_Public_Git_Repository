
// Allows executable to find class Main.
package com.mycompany.serverutilities;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

/**
 * Defines InterfaceForProcessing that includes the signature for
 * processingOfMessageReceivedByEndpoint.
 * This interface allows for storing anonymous-method objects of the form
 * processingOfMessageReceivedByEndpoint in key/value pairs of
 * mapOfEndpointsAndProcessings.
 * @version 0.0
 * @author Tom Lever
 */
public class MessageHandler implements HttpHandler {
    
    private final Controller controller;
    
    public MessageHandler(Controller controllerToUse) {
        this.controller = controllerToUse;
    }
    
    // The handle method of the method handler tears off
    // the HTTP encoding layer from the message and reveals a ice-cream
    // related message. The handle method gets the message from the message
    // interface to the controller.
    @Override
    public void handle(HttpExchange t) {
        
        // t is the HTTP-level message, which includes wrapper information and
        // an ice-cream related message.
        // TODO: Take t, pull out things controller needs, and pass them to
        // controller.process method.
        
        this.controller.process();
        
    }
    
    //public void processingOfMessageReceivedByEndpoint();
}