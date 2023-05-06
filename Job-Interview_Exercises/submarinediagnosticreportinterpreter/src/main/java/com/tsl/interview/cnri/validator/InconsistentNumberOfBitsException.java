/*
 * InconsistentNumberOfBitsException
 * author Tom Lever
 * version 0.0
 * since 01/26/22
 */

package com.tsl.interview.cnri.validator;

public class InconsistentNumberOfBitsException extends Exception {
    
    public InconsistentNumberOfBitsException(String errorMessage) {
        super(errorMessage);
    }
}