
// Allows executable to find class Main.
package com.mycompany.serverutilities;

/**
 * Defines class MessageInterface that allows creation of a message interface
 * based on an inputted endpoint and method of processing HTTP requests from
 * clients and providing HTTP responses to clients.
 * @version 0.0
 * @author Tom Lever
 */
public class MessageInterface {
    
    private final String endpoint;
    private final MessageHandler processing;
    
    public MessageInterface(
            String endpointToUse, MessageHandler processingToUse) {
        this.endpoint = endpointToUse;
        this.processing = processingToUse;
    }
    
    public String getEndpoint() {
        return this.endpoint;
    }
    
    public MessageHandler getProcessing() {
        return this.processing;
    }
}
