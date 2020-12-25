
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
public class InterfaceForProcessing implements HttpHandler {
    @Override
    public void handle(HttpExchange t) {
        
    }
    
    //public void processingOfMessageReceivedByEndpoint();
}