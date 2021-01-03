
/*
 * Distinguishes this design class from MessageInterface in package
 * com.mycompany.serverutilities.
 */
package com.mycompany.designclasses;

/**
 * Defines class MessageInterface that encapsulates functionality to store and
 * get an endpoint and a messageHandler.
 * @version 0.0
 * @author Tom Lever
 */
public class MessageInterface {
    
    // Commented out to allow building.
    //private final String endpoint;
    //private final MessageHandler messageHandler;
    
    /**
     * Defines constructor MessageInterface which sets attributes of this
     * message interface with inputs.
     * @param endpointToUse
     * @param messageHandlerToUse
     */
    // Commented out to allow building.
    //public MessageInterface(
    //    String endpointToUse, MessageHandler messageHandlerToUse) { }
    
    /**
     * Defines method getEndpoint which allows a calling method to get a
     * endpoint stored as an attribute of this MessageInterface object.
     * @return this.endpoint
     */
    public String getEndpoint() {
        // Functionality to get endpoint.
        return new String();
    }

    /**
     * Defines method getMessageHandler which allows a calling method to get a
     * messageHandler stored as an attribute of this MessageInterface object.
     * @return this.messageHandler
     */
    public MessageHandler getMessageHandler() {
        // Functionality to get messageHandler.
        // Commented out to allow building.
        return new MessageHandler(/*new Controller()*/);
    }
}
