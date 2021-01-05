
// Allows executable to find class IceCreamApplicationMessage.
package com.mycompany.serverutilities.clientcommunicationutilities;

/**
 * Defines class IceCreamApplicationMessage whose instances represent ice cream
 * application messages.
 * @version 0.0
 * @author Tom Lever
 */
public class IceCreamApplicationMessage {
    
    private final String bodyOfClientMessage;
    
    /**
     * Defines constructor IceCreamApplicationMethod.
     * @param bodyOfClientMessageToUse
     */
    public IceCreamApplicationMessage(String bodyOfClientMessageToUse) {
        System.out.println("IceCreamApplicationMessage constructor: Started.");
        
        this.bodyOfClientMessage = bodyOfClientMessageToUse;
        System.out.println(
            "IceCreamClientCommunication constructor: Set " +
            "bodyOfClientMessage as bodyOfClientMessageToUse.");
    } 
}
