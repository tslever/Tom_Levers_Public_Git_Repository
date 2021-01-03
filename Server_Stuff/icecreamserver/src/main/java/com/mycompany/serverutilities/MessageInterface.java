
// Allows executable to find class MessageInterface.
package com.mycompany.serverutilities;

/**
 * Defines class MessageInterface that encapsulates functionality to store and
 * get an endpoint and a messageHandler.
 * @version 0.0
 * @author Tom Lever
 */
public class MessageInterface {
    
    private final String endpoint;
    private final MessageHandler messageHandler;
    
    /**
     * Defines constructor MessageInterface which sets attributes of this
     * message interface with inputs.
     * @param endpointToUse
     * @param messageHandlerToUse
     */
    public MessageInterface(
        String endpointToUse, MessageHandler messageHandlerToUse) {
        System.out.println("MessageInterface constructor: Started.");
        
        this.endpoint = endpointToUse;
        this.messageHandler = messageHandlerToUse;
        System.out.println(
            "MessageInterface constructor: Set endpoint as endpointToUse " +
            "and set messageHandler as messageHandlerToUse.");
    }
    
    /**
     * Defines method getEndpoint which allows a calling method to get a
     * endpoint stored as an attribute of this MessageInterface object.
     * @return this.endpoint
     */
    public String getEndpoint() {
        System.out.println("MessageInterface.getEndpoint: Started.");
        
        System.out.println(
            "MessageInterface.getEndpoint: Returning this.endpoint.");
        return this.endpoint;
    }

    /**
     * Defines method getMessageHandler which allows a calling method to get a
     * messageHandler stored as an attribute of this MessageInterface object.
     * @return this.messageHandler
     */
    public MessageHandler getMessageHandler() {
        System.out.println("MessageInterface.getProcessing: Started.");
        
        System.out.println(
            "MessageInterface.getProcessing: Returning this.processing.");
        return this.messageHandler;
    }
}
