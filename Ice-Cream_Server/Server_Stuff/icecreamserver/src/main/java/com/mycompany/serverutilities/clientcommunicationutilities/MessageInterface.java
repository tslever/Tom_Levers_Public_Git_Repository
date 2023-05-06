
// Allows setMessageInterfaces to find class MessageInterface.
package com.mycompany.serverutilities.clientcommunicationutilities;

/**
 * Defines class MessageInterface that encapsulates functionality to store and
 * get an endpoint and a messageHandler.
 * @version 0.0
 * @author Tom Lever
 */
class MessageInterface {
    
    private final String endpoint;
    private final MessageHandler messageHandler;
    
    /**
     * Defines constructor MessageInterface which sets this.endpoint as
     * endpointToUse and this.messageHandler as messageHandlerToUse.
     * @param endpointToUse
     * @param messageHandlerToUse
     */
    public MessageInterface(
        String endpointToUse, MessageHandler messageHandlerToUse) {
        
        this.endpoint = endpointToUse;
        this.messageHandler = messageHandlerToUse;
    }
    
    /**
     * Defines method getEndpoint which returns this.endpoint.
     * @return this.endpoint
     */
    public String getEndpoint() {
        return this.endpoint;
    }

    /**
     * Defines method getMessageHandler which returns this.messageHandler.
     * @return this.messageHandler
     */
    public MessageHandler getMessageHandler() {
        return this.messageHandler;
    }
}
