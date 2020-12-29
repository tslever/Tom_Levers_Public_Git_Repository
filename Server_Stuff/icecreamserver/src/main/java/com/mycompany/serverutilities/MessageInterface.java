
// Allows executable to find class MessageInterface.
package com.mycompany.serverutilities;

/**
 * Defines class MessageInterface that encapsulates functionality to store and
 * get an endpoint and get a processing method.
 * @version 0.0
 * @author Tom Lever
 */
public class MessageInterface {
    
    private final String endpoint;
    private final MessageHandler processing;
    
    /**
     * Defines constructor MessageInterface which sets attributes of this
     * message interface with inputs.
     * @param endpointToUse
     * @param processingToUse
     */
    public MessageInterface(
            String endpointToUse, MessageHandler processingToUse) {
        System.out.println("MessageInterface constructor: Started.");
        
        this.endpoint = endpointToUse;
        this.processing = processingToUse;
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
     * Defines method getProcessing which allows a calling method to get a
     * processing method stored as an attribute of this MessageInterface object.
     * @return this.processing
     */
    public MessageHandler getProcessing() {
        System.out.println("MessageInterface.getProcessing: Started.");
        
        System.out.println(
            "MessageInterface.getProcessing: Returning this.processing.");
        return this.processing;
    }
}
