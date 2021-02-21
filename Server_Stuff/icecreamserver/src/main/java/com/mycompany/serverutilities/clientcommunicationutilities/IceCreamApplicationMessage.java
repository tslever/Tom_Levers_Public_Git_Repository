
// Allows handle to get an IceCreamApplicationMessage.
package com.mycompany.serverutilities.clientcommunicationutilities;

/**
 * Defines class IceCreamApplicationMessage whose instances represent ice cream
 * application messages.
 * @version 0.0
 * @author Tom Lever
 */
class IceCreamApplicationMessage {

    private final String bodyOfClientMessage;
    
    /**
     * Defines constructor IceCreamApplicationMethod, which stores
     * bodyOfClientMessageToUse in this.bodyOfClientMessage.
     * @param bodyOfClientMessageToUse
     */
    public IceCreamApplicationMessage(String bodyOfClientMessageToUse) {
        this.bodyOfClientMessage = bodyOfClientMessageToUse;
    }
    
    /**
     * Defines method getBodyOfClientMessage, which returns String
     * this.bodyOfClientMessage.
     * @return 
     */
    public String getBodyOfClientMessage() {
        return this.bodyOfClientMessage;
    }
}
